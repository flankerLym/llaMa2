"""流式抽取 JSONL，不下载完整大文件；本地文件也支持。"""
import argparse, hashlib, json, random
from pathlib import Path
import requests

def records(source):
    if source.startswith('https://'):
        with requests.get(source, stream=True, timeout=(30, 120)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line: yield json.loads(line)
    else:
        with open(source, encoding='utf-8') as f:
            for line in f:
                if line.strip(): yield json.loads(line)

def normalize(row, stage):
    if stage == 'pretrain':
        text = row.get('text', '')
        return {'text': text} if isinstance(text, str) and text.strip() else None
    messages = row.get('conversations', row.get('messages'))
    if messages is None and 'instruction' in row:
        messages = [{'role':'user','content':row['instruction']+'\n'+row.get('input','')},
                    {'role':'assistant','content':row.get('output','')}]
    if not isinstance(messages, list) or not messages: return None
    if any(m.get('role') not in ('system','user','assistant') or not isinstance(m.get('content'),str)
           or m.get('tool_calls') or m.get('tools') for m in messages): return None
    if not any(m['role']=='assistant' and m['content'].strip() for m in messages): return None
    return {'messages':[{'role':m['role'],'content':m['content']} for m in messages]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pretrain-source', default='https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/312afb4/pretrain_t2t_mini.jsonl')
    p.add_argument('--sft-source', default='https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/312afb4/sft_t2t_mini.jsonl')
    p.add_argument('--pretrain-n',type=int,default=20000)
    p.add_argument('--sft-n',type=int,default=5000)
    p.add_argument('--out',default='data')
    a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    manifest={'seed':42, 'sampling':'first N valid unique rows, then deterministic shuffle', 'sources':{}}
    for stage,source,n in [('pretrain',a.pretrain_source,a.pretrain_n),('sft',a.sft_source,a.sft_n)]:
        if n<20: raise ValueError('每阶段至少 20 条，才能划分训练/验证/测试。')
        rows=[]; seen=set()
        for raw in records(source):
            row=normalize(raw,stage)
            if row is None: continue
            key=json.dumps(row,ensure_ascii=False,sort_keys=True)
            if key in seen: continue
            seen.add(key); rows.append(row)
            if len(rows)%1000==0: print(stage,len(rows),flush=True)
            if len(rows)>=n: break
        if len(rows)<n: raise ValueError(f'{stage} 有效样本不足：{len(rows)} < {n}')
        random.Random(42).shuffle(rows); k=max(1,int(n*.05))
        splits={'train':rows[2*k:], 'val':rows[:k], 'test':rows[k:2*k]}
        manifest['sources'][stage]={'source':source,'requested':n,'splits':{}}
        for split,values in splits.items():
            path=out/f'{stage}_{split}.jsonl'
            path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in values),encoding='utf-8')
            manifest['sources'][stage]['splits'][split]={'rows':len(values),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('数据准备完成：',out)
if __name__=='__main__': main()

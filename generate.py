import argparse,hashlib,json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from model import ModelConfig,Transformer
from train import load

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True)
    p.add_argument('--tokenizer',default='tokenizer'); p.add_argument('--prompt',default='你好，请介绍一下自己。')
    p.add_argument('--chat',action='store_true'); p.add_argument('--max-new-tokens',type=int,default=64)
    p.add_argument('--temperature',type=float,default=.7); p.add_argument('--out')
    a=p.parse_args(); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    state=load(a.checkpoint); tok=AutoTokenizer.from_pretrained(a.tokenizer)
    if hashlib.sha256((Path(a.tokenizer)/'tokenizer.json').read_bytes()).hexdigest()!=state['identity']['tokenizer']:
        raise ValueError('分词器与权重不一致')
    model=Transformer(ModelConfig(**state['config'])).to(device); model.load_state_dict(state['model']); model.eval()
    prompt=tok.apply_chat_template([{'role':'user','content':a.prompt}],tokenize=False,add_generation_prompt=True) if a.chat else '<s>'+a.prompt
    ids=tok.encode(prompt,add_special_tokens=False)
    if len(ids)>state['config']['max_seq_len']: raise ValueError('输入超过上下文长度，请缩短 prompt')
    x=torch.tensor([ids],device=device)
    with torch.no_grad(): model(x) # 预热，计时不含模型加载
    if device.type=='cuda': torch.cuda.synchronize()
    start=time.perf_counter()
    y=model.generate(x,stop_id=tok.eos_token_id,max_new_tokens=a.max_new_tokens,temperature=a.temperature,top_k=40)
    if device.type=='cuda': torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    result={'prompt':a.prompt,'answer':tok.decode(y[0].tolist(),skip_special_tokens=True),
            'seconds':elapsed,'generated_tokens':y.size(1),'tokens_per_sec':y.size(1)/elapsed,
            'device':str(device),'note':'无 KV Cache；这是单次生成测量，不是严谨性能基准。'}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if a.out: Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__': main()

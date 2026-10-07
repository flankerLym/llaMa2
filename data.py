"""逐行随机访问；预先过滤截断后没有回答 token 的 SFT 样本。"""
import json
import torch
from torch.utils.data import Dataset

def encode(row,tok,stage,length):
    if stage=='pretrain':
        ids=[tok.bos_token_id]+tok.encode(row['text'],add_special_tokens=False)+[tok.eos_token_id]
        labels=ids.copy()
    else:
        ids=[]; labels=[]
        for m in row['messages']:
            header=tok.encode('<|im_start|>'+m['role']+'\n',add_special_tokens=False)
            body=tok.encode(m['content'],add_special_tokens=False)+[tok.eos_token_id]
            tail=tok.encode('\n',add_special_tokens=False)
            ids+=header+body+tail
            labels+=[-100]*len(header)+(body if m['role']=='assistant' else [-100]*len(body))+[-100]*len(tail)
    ids=ids[:length+1]; labels=labels[:length+1]
    n=len(ids)-1
    x=ids[:-1]+[tok.pad_token_id]*(length-n)
    y=labels[1:]+[-100]*(length-n)
    return torch.tensor(x),torch.tensor(y)

class TextDataset(Dataset):
    def __init__(self,path,tok,stage,length):
        self.path=path; self.tok=tok; self.stage=stage; self.length=length; self.offsets=[]; self.skipped=0
        with open(path,'rb') as f:
            while True:
                offset=f.tell(); line=f.readline()
                if not line: break
                row=json.loads(line)
                if encode(row,tok,stage,length)[1].ne(-100).any(): self.offsets.append(offset)
                else: self.skipped+=1
        if not self.offsets: raise ValueError(f'{path} 无有效样本，请增大 seq_len 或检查数据。')
        print(path,'有效',len(self),'过滤',self.skipped)
    def __len__(self): return len(self.offsets)
    def __getitem__(self,i):
        with open(self.path,'rb') as f:
            f.seek(self.offsets[i]); row=json.loads(f.readline())
        return encode(row,self.tok,self.stage,self.length)

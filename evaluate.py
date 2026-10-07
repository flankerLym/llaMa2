import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from data import TextDataset
from model import ModelConfig,Transformer
from train import load,evaluate

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True)
    p.add_argument('--stage',choices=['pretrain','sft'],required=True)
    p.add_argument('--tokenizer',default='tokenizer'); p.add_argument('--data-dir',default='data')
    p.add_argument('--batch-size',type=int,default=4); p.add_argument('--out',required=True)
    a=p.parse_args(); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    state=load(a.checkpoint); tok=AutoTokenizer.from_pretrained(a.tokenizer)
    import hashlib
    if hashlib.sha256((Path(a.tokenizer)/'tokenizer.json').read_bytes()).hexdigest()!=state['identity']['tokenizer']:
        raise ValueError('分词器与权重不一致')
    model=Transformer(ModelConfig(**state['config'])).to(device); model.load_state_dict(state['model'])
    ds=TextDataset(str(Path(a.data_dir)/f'{a.stage}_test.jsonl'),tok,a.stage,state['config']['max_seq_len'])
    result=evaluate(model,DataLoader(ds,batch_size=a.batch_size),device)
    result.update(checkpoint=a.checkpoint,stage=a.stage,split='test',effective_rows=len(ds),skipped_rows=ds.skipped)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(result,indent=2))
    print(result)
if __name__=='__main__': main()

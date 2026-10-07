"""验证结构、因果掩码和 SFT 标签；无需下载数据。"""
import torch
from transformers import AutoTokenizer
from model import ModelConfig,Transformer
from data import encode
import argparse
p=argparse.ArgumentParser(); p.add_argument('--tokenizer',default='tokenizer'); a=p.parse_args()
torch.set_num_threads(2); torch.manual_seed(42)
tok=AutoTokenizer.from_pretrained(a.tokenizer)
m=Transformer(ModelConfig(dim=64,n_layers=2,n_heads=4,n_kv_heads=2,vocab_size=len(tok),max_seq_len=128,multiple_of=32)).eval()
x=torch.randint(6,len(tok),(1,12)); z=x.clone(); z[:,6:]=torch.randint(6,len(tok),(1,6))
with torch.no_grad():
    first=m(x,targets=x).logits.clone(); second=m(z,targets=z).logits.clone()
assert torch.allclose(first[:,:6],second[:,:6],atol=1e-6), '因果掩码错误'
row={'messages':[{'role':'user','content':'你好'},{'role':'assistant','content':'世界'}]}
x,y=encode(row,tok,'sft',128)
assert y[y.ne(-100)].tolist()==tok.encode('世界',add_special_tokens=False)+[tok.eos_token_id], 'SFT标签错误'
expected=tok.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=False)
assert x[:len(expected)-1].tolist()==expected[:-1], '训练模板与推理模板不一致'
assert m.tok_embeddings.weight is m.output.weight
assert torch.isfinite(m(x[None],targets=y[None]).logits).all()
print('PASS: 因果注意力、SFT回答标签、模板一致性、权重共享、前向计算')

"""单 GPU/CPU 训练：FP16、梯度累积、裁剪、验证、原子保存、同配置续训。"""
import argparse, contextlib, csv, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import ModelConfig,Transformer
from data import TextDataset

def amp(device):
    return torch.autocast('cuda',dtype=torch.float16) if device.type=='cuda' else contextlib.nullcontext()
def loss_sum(model,x,y):
    logits=model(x,targets=y).logits
    return F.cross_entropy(logits.float().reshape(-1,logits.size(-1)),y.reshape(-1),ignore_index=-100,reduction='sum')
@torch.no_grad()
def evaluate(model,loader,device,limit=0):
    model.eval(); total=0.; count=0
    for i,(x,y) in enumerate(loader):
        if limit and i>=limit: break
        x,y=x.to(device),y.to(device)
        with amp(device): total+=loss_sum(model,x,y).item()
        count+=y.ne(-100).sum().item()
    model.train()
    value=total/count
    return {'loss':value,'ppl':math.exp(min(value,80)),'tokens':count,'batches_limit':limit}
def save(path,state):
    temp=path.with_suffix('.tmp'); torch.save(state,temp); temp.replace(path)
def load(path): return torch.load(path,map_location='cpu',weights_only=False) # 仅加载自己生成的 checkpoint

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['pretrain','sft'],required=True)
    p.add_argument('--config',default='configs/small.json'); p.add_argument('--data-dir',default='data')
    p.add_argument('--tokenizer',default='tokenizer'); p.add_argument('--out',default=None)
    p.add_argument('--init'); p.add_argument('--resume'); p.add_argument('--steps',type=int,default=500)
    p.add_argument('--batch-size',type=int,default=4); p.add_argument('--accum',type=int,default=4)
    p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--eval-batches',type=int,default=20); p.add_argument('--cpu',action='store_true')
    a=p.parse_args()
    if min(a.steps,a.batch_size,a.accum,a.eval_every)<1: raise ValueError('步数/batch/accum/eval-every 必须为正数')
    if a.init and a.resume: raise ValueError('init 与 resume 不能同时使用')
    if a.stage=='sft' and not (a.init or a.resume): raise ValueError('SFT 必须 --init 预训练权重 或 --resume SFT断点')
    torch.set_num_threads(2); random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device=torch.device('cuda:0' if torch.cuda.is_available() and not a.cpu else 'cpu')
    tok=AutoTokenizer.from_pretrained(a.tokenizer); cfg=json.loads(Path(a.config).read_text())
    cfg.update(vocab_size=len(tok),pad_token_id=tok.pad_token_id)
    model=Transformer(ModelConfig(**cfg)).to(device)
    out=Path(a.out or f'outputs/{a.stage}'); out.mkdir(parents=True,exist_ok=True)
    train_path=Path(a.data_dir)/f'{a.stage}_train.jsonl'
    identity={'data':hashlib.sha256(train_path.read_bytes()).hexdigest(),
              'tokenizer':hashlib.sha256((Path(a.tokenizer)/'tokenizer.json').read_bytes()).hexdigest()}
    settings={'steps':a.steps,'batch_size':a.batch_size,'accum':a.accum,'lr':a.lr}
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.1)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    start=0; best=float('inf'); state=None
    if a.init or a.resume:
        state=load(a.resume or a.init)
        if state['config']!=cfg or state['identity']['tokenizer']!=identity['tokenizer']:
            raise ValueError('模型配置/分词器与checkpoint不一致。不要重训分词器后继续用旧权重。')
        model.load_state_dict(state['model'])
    if a.resume:
        if state['stage']!=a.stage or state['identity']!=identity or state['settings']!=settings:
            raise ValueError('续训须保持阶段、数据和训练参数相同；延长训练请用 --init 进行新一轮训练。')
        optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        start=state['step']; best=state['best']
        torch.set_rng_state(state['rng']); random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        if device.type=='cuda' and state['cuda_rng'] is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
    train=TextDataset(str(train_path),tok,a.stage,cfg['max_seq_len'])
    val=TextDataset(str(Path(a.data_dir)/f'{a.stage}_val.jsonl'),tok,a.stage,cfg['max_seq_len'])
    val_loader=DataLoader(val,batch_size=a.batch_size)
    # 按独立 seed 产生每个 epoch 的排列，续训从精确的 batch 位置开始。
    batches=math.ceil(len(train)/a.batch_size); current_epoch=-1; iterator=None
    def next_batch(micro):
        nonlocal current_epoch,iterator
        epoch,pos=divmod(micro,batches)
        if epoch!=current_epoch:
            gen=torch.Generator().manual_seed(42+epoch)
            order=torch.randperm(len(train),generator=gen).tolist()
            loader=DataLoader(train,batch_size=a.batch_size,sampler=order,num_workers=0,
                              generator=torch.Generator().manual_seed(123+epoch))
            iterator=iter(loader); current_epoch=epoch
            for _ in range(pos): next(iterator)
        return next(iterator)
    print('device:',device,'参数量:',sum(p.numel() for p in model.parameters())/1e6,'M',flush=True)
    baseline=evaluate(model,val_loader,device,a.eval_batches)
    (out/f'baseline_{start}.json').write_text(json.dumps(baseline,indent=2))
    print('训练前验证：',baseline,flush=True)
    log=out/'metrics.csv'; mode='a' if a.resume and log.exists() else 'w'
    with log.open(mode,newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['step','train_loss','val_loss','val_ppl','tokens_per_sec','peak_gpu_gb'])
        if mode=='w': writer.writeheader()
        for step in range(start,a.steps):
            model.train(); optimizer.zero_grad(set_to_none=True); started=time.perf_counter()
            if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
            group=[next_batch(step*a.accum+j) for j in range(a.accum)]
            count=sum(y.ne(-100).sum().item() for _,y in group); total=0.
            warmup=min(50,max(1,a.steps//10))
            factor=(step+1)/warmup if step<warmup else .1+.9*(1+math.cos(math.pi*(step-warmup)/max(1,a.steps-warmup)))/2
            for g in optimizer.param_groups: g['lr']=a.lr*factor
            for x,y in group:
                x,y=x.to(device),y.to(device)
                with amp(device): loss=loss_sum(model,x,y); objective=loss/count
                total+=loss.item(); scaler.scale(objective).backward()
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer); scaler.update()
            if device.type=='cuda': torch.cuda.synchronize()
            speed=count/(time.perf_counter()-started)
            if (step+1)%10==0: print(f'{a.stage} step={step+1}/{a.steps} loss={total/count:.4f} tokens/s={speed:.0f}',flush=True)
            if (step+1)%a.eval_every==0 or step+1==a.steps:
                metrics=evaluate(model,val_loader,device,a.eval_batches)
                improved=metrics['loss']<best; best=min(best,metrics['loss'])
                checkpoint={'model':model.state_dict(),'config':cfg,'optimizer':optimizer.state_dict(),
                    'scaler':scaler.state_dict(),'step':step+1,'best':best,'stage':a.stage,'identity':identity,
                    'settings':settings,'rng':torch.get_rng_state(),'python_rng':random.getstate(),
                    'numpy_rng':np.random.get_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}
                save(out/'last.pt',checkpoint)
                if improved: save(out/'best.pt',checkpoint)
                writer.writerow(dict(step=step+1,train_loss=total/count,val_loss=metrics['loss'],val_ppl=metrics['ppl'],
                    tokens_per_sec=speed,peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9 if device.type=='cuda' else 0))
                f.flush(); print('验证：',metrics,'已保存',out/'last.pt',flush=True)
if __name__=='__main__': main()

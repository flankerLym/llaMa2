import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(10,4))
for stage in ['pretrain','sft']:
    path=Path('outputs')/stage/'metrics.csv'
    if not path.exists(): continue
    rows=list(csv.DictReader(path.open()))
    if not rows: continue
    x=[int(r['step']) for r in rows]
    axes[0].plot(x,[float(r['train_loss']) for r in rows],label=stage+' train')
    axes[0].plot(x,[float(r['val_loss']) for r in rows],label=stage+' val')
    axes[1].plot(x,[float(r['val_ppl']) for r in rows],label=stage)
for ax,title in zip(axes,['Token loss','Validation perplexity']):
    ax.set_title(title); ax.set_xlabel('Optimizer steps'); ax.legend(); ax.grid(alpha=.3)
Path('outputs').mkdir(exist_ok=True); fig.tight_layout(); fig.savefig('outputs/loss.png',dpi=160)

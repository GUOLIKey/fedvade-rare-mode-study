"""Isolated synthetic integration checks (2 pretrain + 3 rounds, not research runs)."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);a=p.parse_args()
scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
from common import train as T
from common.evaluate import evaluate_run
root=a.out.resolve();root.mkdir(parents=True,exist_ok=False)
data=root/'arbitrary_input_location';(data/'cache').mkdir(parents=True)
rng=np.random.RandomState(123)
y=np.repeat([0,1],120);X=(rng.randn(240,4)*.25+y[:,None]*2).astype(np.float32)
np.savez_compressed(data/'cache/toy.npz',X=X,y=y)
variants=[('author_a',[]),('author_b',[]),('epoch',['--schedule','epoch']),
          ('dp',['--local-gmm','dpgmm']),('samek',['--local-gmm','samek_gmm']),
          ('reweight',['--sample-floor','1','--usage-floor','.1'])]
report={}
for name,extra in variants:
    out=root/name
    args=T.build_parser().parse_args(['--source',str(a.source),'--out',str(out),
        '--data-dir',str(data),'--dataset','toy','--mode','clusterlet','--device','cpu',
        '--clients','2','--batch-size','16','--hid-dim','2','--sample','20',
        '--pretrain','2','--rounds','3','--save-every','3','--seed','7',*extra])
    T.run(args)
    cfg=json.loads((out/'config.json').read_text())
    result=json.loads((out/'result.json').read_text())
    assert result['status']=='ok' and result['nan_events']==0
    assert Path(cfg['imported_model_file'])==out/'source/model.py'
    evaluated=evaluate_run(out,'toy','clusterlet',2,rounds=(3,),data_dir=data,
                           population='unique_full',output=root/(name+'_evaluation.json'))
    report[name]={'status':result['status'],'client_lr':result['last']['client_lr'],
                  'retained_K':cfg.get('retained_K'),'n_eval':evaluated['n_evaluation']}
aa=torch.load(root/'author_a/last_checkpoint.pt',map_location='cpu',weights_only=False)['server']
bb=torch.load(root/'author_b/last_checkpoint.pt',map_location='cpu',weights_only=False)['server']
assert all(torch.equal(aa[k],bb[k]) for k in aa)
assert report['dp']['retained_K']==report['samek']['retained_K']
assert report['epoch']['client_lr']!=report['author_a']['client_lr']
report['repeated_author_weights_equal']=True
env=os.environ.copy();env['PYTHONPATH']=str(scripts)
subprocess.run([sys.executable,str(scripts/'common/evaluate.py'),
    '--run',str(root/'author_a'),'--data-dir',str(data),'--rounds','3',
    '--output',str(root/'standalone_evaluation.json')],check=True,env=env)
subprocess.run([sys.executable,str(scripts/'lines/coverage_metric/main.py'),
    '--run',str(root/'author_a'),'--data-dir',str(data),'--self-check',
    '--output',str(root/'standalone_coverage_selfcheck.json')],check=True,env=env)
(root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

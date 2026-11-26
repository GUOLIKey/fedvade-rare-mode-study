"""Reevaluate historical weights without editing runs or training models."""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits

p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);a=p.parse_args()
root=a.project.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
from common import data as D
from common.evaluate import evaluate_run
from common.provenance import array_sha256, sha256
base=root/'experiments/fmnist_clusterlets_v1';data=base/'data'
torch.set_num_threads(1);torch.set_num_interop_threads(1)
summary={'partitions':[],'evaluations':[],'coverage':[]}
with threadpool_limits(limits=1):
    # Read the unchanged historical partition implementation in an isolated namespace.
    sys.path.insert(0,str(data))
    sys.path.insert(0,str(root/'fuxian-round2/FedVaDE_source'))
    spec=importlib.util.spec_from_file_location('historical_fm_data',data/'fm_data.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    for group,dataset,mode in [('bal_clusterlet','fmnist_bal','clusterlet'),
                                ('lt10_clusterlet','fmnist_lt10','clusterlet'),
                                ('lt10_author','fmnist_lt10','author')]:
        run=base/'runs'/group/'s0';cfg=json.loads((run/'config.json').read_text())
        before=old.make_clients(dataset,mode,10,200,data,run/'source',seed=42)
        after=D.make_clients(dataset,mode,10,200,data,run/'source',seed=42)
        same=torch.equal(before[2][0],after[2][0]) and torch.equal(before[2][1],after[2][1])
        same=same and all(torch.equal(x,y) for aa,bb in zip(before[1],after[1])
                         for x,y in zip(aa.dataset.tensors,bb.dataset.tensors))
        assert same, 'partition/data regression: '+group
        sizes=after[3]
        # mini-batch steps per client, not a guessed global 4.6x LR multiplier
        summary['partitions'].append({'group':group,'historical_tensors_equal':same,
            'input_sha256':array_sha256(after[2][0].numpy()),'labels_sha256':array_sha256(after[2][1].numpy()),
            'client_sizes':sizes,'batches_per_client_per_round':[(n+199)//200 for n in sizes],
            'total_client_records':sum(sizes),'n_input':len(after[2][1])})
        for seed in (0,1,2):
            run=base/'runs'/group/('s%d'%seed)
            historical=json.loads((run/'metrics.json').read_text())['evaluated_rounds']['300']
            for population in ('legacy_clients','unique_full'):
                result=evaluate_run(run,dataset,mode,10,rounds=(300,),data_dir=data,
                    population=population,output=out/('%s_s%d_%s.json'%(group,seed,population)))
                val=result['rounds']['300']['ari']['mean']
                delta=val-historical['ari_mean']
                if population=='legacy_clients':
                    assert abs(delta)<1e-10, (group,seed,'legacy ARI drift',delta)
                summary['evaluations'].append({'group':group,'seed':seed,'population':population,
                    'n_evaluation':result['n_evaluation'],'ari':val,'historical_ari':historical['ari_mean'],
                    'delta_from_historical':delta})
            # Three generation seeds at fixed reference split; additional split sensitivity on s0.
            target=out/('%s_s%d_coverage.json'%(group,seed))
            cmd=[sys.executable,str(scripts/'lines/coverage_metric/main.py'),
                 '--run',str(run),'--data-dir',str(data),'--output',str(target),
                 '--generation-seeds','0','1','2','--split-seeds','0']
            subprocess.run(cmd,check=True)
            coverage=json.loads(target.read_text())
            old_cov=json.loads((run/'coverage_v2.json').read_text())
            assert abs(coverage['draws'][0]['overall_ratio']-old_cov['overall_ratio'])<6e-5
            summary['coverage'].append({'group':group,'seed':seed,
                'historical_ratio':old_cov['overall_ratio'],'generation_mean':coverage['overall_mean'],
                'generation_sd':coverage['overall_sd_ddof0']})
    # Width groups: saved-model generation sensitivity only, no new training.
    for width in (20,30):
        for seed in (0,1,2):
            run=base/'runs_hiddim'/('hid%d'%width)/('s%d'%seed)
            target=out/('hid%d_s%d_coverage.json'%(width,seed))
            subprocess.run([sys.executable,str(scripts/'lines/coverage_metric/main.py'),
                '--run',str(run),'--data-dir',str(data),'--output',str(target),
                '--generation-seeds','0','1','2','--split-seeds','0'],check=True)
            cov=json.loads(target.read_text())
            summary['coverage'].append({'group':'hid%d'%width,'seed':seed,
                'generation_mean':cov['overall_mean'],'generation_sd':cov['overall_sd_ddof0']})
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'evaluations':len(summary['evaluations']),'coverage_runs':len(summary['coverage'])}))

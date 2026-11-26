"""Real-data missing-class controls; no model training or historical writes."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
from common import data as D
from common.provenance import array_sha256,write_new_json
spec=importlib.util.spec_from_file_location('coverage_controls',scripts/'lines/coverage_metric/main.py')
cov=importlib.util.module_from_spec(spec);spec.loader.exec_module(cov)
result={'scope':'real reference-set class deletion, equal total sample count; not a learned generator',
        'split_seed':0,'replacement_seed':0,'datasets':[]}
with threadpool_limits(limits=1):
 for dataset in ('fmnist_bal','fmnist_lt10'):
    X,y=D.load_dataset(a.data_dir,dataset);X,_=D.standardize(X)
    ref,qry=D.half_split(y,0);R,Q,Yq=X[ref],X[qry],y[qry]
    identity=cov.ratios(R,Q,Yq,R)
    rows=[]
    for c in (1,5,8,9):
        available=R[y[ref]!=c]
        n_missing=len(R)-len(available)
        rng=np.random.RandomState(0)
        G=np.concatenate([available,available[rng.choice(len(available),n_missing,replace=True)]])
        sel=Yq==c
        value=cov.ratios(R,Q[sel],Yq[sel],G)
        rows.append({'deleted_class':c,'removed_reference_count':n_missing,
                     'n_ref':len(R),'n_gen':len(G),'ratio':value['overall_ratio']})
    result['datasets'].append({'dataset':dataset,'input_sha256':array_sha256(X),
        'identity':identity,'deletion_controls':rows})
write_new_json(a.output,result)
print(json.dumps(result,indent=2))

"""Size-matched nearest-neighbour distance ratios, with limited validation controls.

This statistic does not prove distributional equivalence or absence of mode loss.
Generation seeds and reference-split seeds are separate from prediction seeds.
Outputs must use a new path, never overwrite historical results.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from common import data as D, distances as DIST, model as M, sampling as S
from common.provenance import resolve_data_dir, write_new_json, array_sha256, sha256, script_fingerprint, runtime_versions


def ratios(R,Q,Yq,G):
    if len(G)!=len(R):
        raise ValueError('size-matched metric requires len(G) == len(R)')
    d_real=DIST.min_dist_to_set(Q,R); d_gen=DIST.min_dist_to_set(Q,G)
    rows=[]
    for c in sorted(np.unique(Yq)):
        sel=Yq==c
        mr,mg=float(np.median(d_real[sel])),float(np.median(d_gen[sel]))
        if mr<=0:
            raise ValueError('zero reference distance; ratio is undefined')
        rows.append({'class':int(c),'n_query':int(sel.sum()),'median_d_real':mr,
                     'median_d_gen':mg,'ratio':mg/mr})
    den=float(np.median(d_real))
    if den<=0: raise ValueError('zero overall reference distance')
    return {'overall_ratio':float(np.median(d_gen)/den),'per_class':rows}


def metric_controls():
    R=np.array([[-10.,0.],[-10.,1.],[10.,0.],[10.,1.]],np.float32)
    Q=np.array([[-10.,.25],[-10.,.75],[10.,.25],[10.,.75]],np.float32)
    labels=np.array([0,0,1,1])
    identity=ratios(R,Q,labels,R)
    missing=ratios(R,Q,labels,np.concatenate([R[:2],R[:2]]))
    result={'identity_pass':all(abs(r['ratio']-1)<1e-6 for r in identity['per_class']),
            'missing_class_pass':missing['per_class'][1]['ratio']>10,
            'missing_class_ratio':missing['per_class'][1]['ratio'],
            'scope':'synthetic separated classes only; not full validation on real data'}
    if not result['identity_pass'] or not result['missing_class_pass']:
        raise RuntimeError('metric validation control failed')
    return result


def prepare(run,data_dir):
    cfg=json.loads((run/'config.json').read_text())
    data_dir=resolve_data_dir(cfg,data_dir)
    _,_,(X,y),_,_=D.make_clients(cfg['dataset'],cfg['partition_mode'],cfg['clients'],
        cfg['batch_size'],data_dir,run/'source')
    return cfg,X.numpy(),y.numpy(),data_dir


def compute(run,data_dir,n_gen,split_seed,self_check,generation_seed=0):
    cfg,X,Y,data_dir=prepare(run,data_dir)
    return _compute(run,cfg,X,Y,n_gen,split_seed,self_check,generation_seed)


def _compute(run,cfg,X,Y,n_gen,split_seed,self_check,generation_seed,model=None):
    ref,qry=D.half_split(Y,seed=split_seed)
    R,Q,Yq=X[ref],X[qry],Y[qry]
    if n_gen is not None and n_gen!=len(R):
        raise ValueError('--n-gen must equal reference count %d'%len(R))
    if self_check:
        G=R
    else:
        if model is None:
            state,_,_=M.load_server_state(run)
            model=M.build_vade(state,M.hid_dim_of(cfg),cfg['alpha'],X.shape[1],source=run/'source')
        G=S.generate(model,len(R),seed=generation_seed)
    value=ratios(R,Q,Yq,G)
    value.update(n_ref=len(R),n_query=len(Q),n_gen=len(G),split_seed=split_seed,
                 generation_seed=generation_seed,self_check=self_check)
    if self_check and (abs(value['overall_ratio']-1)>1e-6 or
                       any(abs(r['ratio']-1)>1e-6 for r in value['per_class'])):
        raise RuntimeError('identity check failed')
    return value


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path)
    ap.add_argument('--data-dir',type=Path)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--n-gen',type=int)
    ap.add_argument('--generation-seeds',type=int,nargs='+',default=[0,1,2])
    ap.add_argument('--split-seeds',type=int,nargs='+',default=[0])
    ap.add_argument('--self-check',action='store_true')
    ap.add_argument('--metric-controls',action='store_true')
    a=ap.parse_args(); package=script_fingerprint(Path(__file__).resolve().parents[2])
    controls=metric_controls()
    if a.metric_controls:
        print(json.dumps(controls,indent=2)); return
    if a.run is None or a.output is None:
        ap.error('--run and --output are required for saved-weight evaluation')
    run=a.run.resolve(); cfg,X,Y,data_dir=prepare(run,a.data_dir)
    model=None
    if not a.self_check:
        state,_,_=M.load_server_state(run)
        model=M.build_vade(state,M.hid_dim_of(cfg),cfg['alpha'],X.shape[1],source=run/'source')
    draws=[_compute(run,cfg,X,Y,a.n_gen,split,a.self_check,seed,model)
           for split in a.split_seeds for seed in ([0] if a.self_check else a.generation_seeds)]
    result={'schema':'coverage_audit_v1','run':str(run),'training_seed':cfg.get('seed'),
            'repair_package':package,'runtime_packages':runtime_versions(),
            'generation_seeds':a.generation_seeds,'split_seeds':a.split_seeds,
            'input_sha256':array_sha256(X),'label_sha256':array_sha256(Y),
            'checkpoint_sha256':None if a.self_check else sha256(run/'last_checkpoint.pt'),
            'data_dir':str(data_dir),'controls':controls,'draws':draws,
            'overall_mean':float(np.mean([d['overall_ratio'] for d in draws])),
            'overall_sd_ddof0':float(np.std([d['overall_ratio'] for d in draws])),
            'note':'Within one training run; no pooling with between-training-seed uncertainty.'}
    write_new_json(a.output,result)
    print(json.dumps({'output':str(a.output),'overall_mean':result['overall_mean'],'draws':len(draws)}))

if __name__=='__main__': main()

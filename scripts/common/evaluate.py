"""Explicit, non-overwriting saved-weight evaluation; no training."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'common'

import numpy as np
from . import data as D, metrics as MET, model as M
from .provenance import resolve_data_dir, write_new_json, array_sha256, source_hashes, sha256, script_fingerprint, runtime_versions

ROUNDS = (100, 300)


def evaluate_run(run: Path, dataset: str, mode: str, n_inference: int = 10,
                 rounds=ROUNDS, data_dir=None, population='unique_full', output=None,
                 checkpoint=None):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    package = script_fingerprint(Path(__file__).resolve().parents[1])
    run = Path(run).resolve()
    cfg = json.loads((run/'config.json').read_text())
    data_dir = resolve_data_dir(cfg, data_dir)
    if population not in ('unique_full', 'legacy_clients'):
        raise ValueError('unknown evaluation population')
    if n_inference < 1:
        raise ValueError('at least one inference seed is required')
    k, loaders, (X, y), sizes, _ = D.make_clients(
        dataset, mode, cfg['clients'], cfg['batch_size'], data_dir, run/'source')
    evaluated = loaders if population == 'legacy_clients' else [
        DataLoader(TensorDataset(X,y),batch_size=cfg['batch_size'],shuffle=False)]
    out = {'schema':'audit_evaluation_v1','run':str(run),'dataset':dataset,'mode':mode,
           'repair_package':package, 'runtime_packages':runtime_versions(),
           'training_seed_from_config':cfg.get('seed'),
           'environment':{'torch':torch.__version__,'numpy':np.__version__,'torch_threads':torch.get_num_threads()},
           'population':population,'independent_test':False,'n_evaluation':sum(len(ld.dataset) for ld in evaluated),
           'n_client_records':sum(sizes),'n_unique_input':len(y),
           'input_sha256':array_sha256(X.numpy()),'label_sha256':array_sha256(y.numpy()),
           'source_hashes':source_hashes(run/'source'),'data_dir':str(data_dir),
           'inference_seeds':list(range(n_inference)),'sd_ddof':0,'rounds':{}}
    requested = [(str(rnd),run/('epoch_%03d_server.pt'%rnd)) for rnd in rounds]
    if checkpoint:
        requested = [('checkpoint', Path(checkpoint).resolve())]
    for label, ck in requested:
        final_fallback = False
        if not ck.is_file():
            final = run/'last_checkpoint.pt'
            if checkpoint or not final.is_file():
                continue
            ck, final_fallback = final, True
        state, payload, _ = M.load_server_state(run, str(ck))
        if final_fallback and payload.get('round') != int(label):
            continue  # Never present the final model as a different requested round.
        model = M.build_vade(state,M.hid_dim_of(cfg),cfg['alpha'],X.shape[1],source=run/'source')
        draws=[]
        for seed in range(n_inference):
            torch.manual_seed(seed)
            preds, labels=[],[]
            with torch.no_grad():
                for ld in evaluated:
                    for xb,yb in ld:
                        preds.append(model.predict(xb)); labels.append(yb.numpy())
            draws.append(MET.cluster_metrics(np.concatenate(labels),np.concatenate(preds)))
        out['rounds'][label] = {key:{'mean':float(np.mean([d[key] for d in draws])),
            'sd':float(np.std([d[key] for d in draws]))} for key in ('ari','purity','nmi','acc')}
        out['rounds'][label].update(draws=draws,checkpoint=str(ck),checkpoint_sha256=sha256(ck),
                                    round=payload.get('round'))
    if not out['rounds']:
        raise FileNotFoundError('no requested checkpoint was found; nothing evaluated')
    write_new_json(output or run/'metrics.json',out)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data-dir',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--population',choices=('unique_full','legacy_clients'),default='unique_full')
    p.add_argument('--inference-seeds',type=int,default=10)
    p.add_argument('--rounds',type=int,nargs='+',default=list(ROUNDS))
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--dataset',help='required when absent from a historical config')
    p.add_argument('--mode',choices=('clusterlet','author'),help='required when absent from a historical config')
    a=p.parse_args(); cfg=json.loads((a.run/'config.json').read_text())
    dataset, mode = a.dataset or cfg.get('dataset'), a.mode or cfg.get('partition_mode')
    if dataset is None or mode is None:
        p.error('historical config is incomplete; pass --dataset and --mode explicitly')
    result=evaluate_run(a.run,dataset,mode,a.inference_seeds,
        a.rounds,a.data_dir,a.population,a.output,a.checkpoint)
    print(json.dumps({'population':result['population'],'n':result['n_evaluation'],
                      'output':str(a.output)}))

if __name__=='__main__': main()

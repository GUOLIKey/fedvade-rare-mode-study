"""Global clustering metrics for a finished run (threshold-free, shared by lines)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import data as D
from . import metrics as MET
from . import model as M

ROUNDS = (100, 300)


def evaluate_run(run: Path, dataset: str, mode: str, n_inference: int = 10,
                 rounds=ROUNDS, data_dir=None) -> dict:
    """Frozen model, N inference seeds, ARI/Purity/NMI per saved round."""
    import torch
    run = Path(run)
    cfg = json.loads((run / "config.json").read_text())
    data_dir = Path(data_dir) if data_dir else run.parents[2] / "data"
    k, loaders, (X, y), sizes, _ = D.make_clients(
        dataset, mode, cfg["clients"], cfg["batch_size"], data_dir, run / "source")
    out = {"run": str(run), "dataset": dataset, "mode": mode, "rounds": {}}
    for rnd in rounds:
        ck = run / ("epoch_%03d_server.pt" % rnd)
        if not ck.exists():
            continue
        state, _, _ = M.load_server_state(run, ck.name)
        model = M.build_vade(state, M.hid_dim_of(cfg), cfg["alpha"], X.shape[1])
        draws = []
        for s in range(n_inference):
            torch.manual_seed(s)
            preds, labels = [], []
            with torch.no_grad():
                for ld in loaders:
                    for xb, yb in ld:
                        preds.append(model.predict(xb))
                        labels.append(yb.numpy())
            draws.append(MET.cluster_metrics(np.concatenate(labels), np.concatenate(preds)))
        out["rounds"][str(rnd)] = {
            key: {"mean": float(np.mean([d[key] for d in draws])),
                  "sd": float(np.std([d[key] for d in draws]))}
            for key in ("ari", "purity", "nmi")}
    (run / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    return out

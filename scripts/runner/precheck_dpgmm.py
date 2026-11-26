"""Step 1: how many components does DPGMM keep per client?

Reads the local_mu tensors dumped after pretraining (runner --dump-pretrain)
and fits DPGMM for several weight_concentration_prior values.  No training.

The number that matters for FedVaDE is the TOTAL across clients, because the
server only samples `sample` latent points per round.  If the total is large,
each component sees ~sample/total points and the rare-component problem of ACD
comes back.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import local_gmm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--priors", nargs="+", type=float, default=[0.01, 0.1, 1.0])
    ap.add_argument("--prune", type=float, default=0.01)
    ap.add_argument("--n-components", type=int, default=15)
    ap.add_argument("--sample", type=int, default=100)
    a = ap.parse_args()

    dump = a.dump.resolve()
    files = sorted(dump.glob("client_*_mu.npy"))
    if not files:
        raise SystemExit("no client_*_mu.npy in %s" % dump)
    meta = json.loads((dump / "meta.json").read_text()) if (dump / "meta.json").exists() else {}
    mus = [np.load(f) for f in files]
    print("=" * 78)
    print("%s  (%s / %s)  clients=%d" % (dump.name, meta.get("dataset"), meta.get("mode"), len(mus)))
    print("每客户端样本数:", [len(m) for m in mus])

    result = {"dump": str(dump), "meta": meta, "priors": {}}
    for wcp in a.priors:
        ks, allw, allc = [], [], []
        for m in mus:
            w, mu, cov = local_gmm.fit_dpgmm(m, wcp, n_components=a.n_components,
                                             prune=a.prune)
            ks.append(len(w))
            allw.append(w)
            allc.append(cov)
        total = int(sum(ks))
        var = np.concatenate([c.reshape(len(c), -1).mean(1) for c in allc])
        result["priors"][str(wcp)] = {
            "retained_K_per_client": [int(k) for k in ks],
            "total_components": total,
            "expected_samples_per_component": round(a.sample / total, 3) if total else None,
            "median_component_variance": round(float(np.median(var)), 6),
            "min_component_variance": round(float(np.min(var)), 8),
        }
        print("\nprior = %-5s  每客户端保留 K = %s" % (wcp, ks))
        print("  总分量数 = %2d   每轮每分量期望样本 = %.2f   方差中位数 = %.3e"
              % (total, a.sample / total if total else 0, float(np.median(var))))

    (dump / "precheck.json").write_text(json.dumps(result, indent=2) + "\n")
    print("\n判据：总分量数 <= 60（每分量 >= 1.7 个点）算可用；> 80 视为 ACD 式爆炸")


if __name__ == "__main__":
    main()

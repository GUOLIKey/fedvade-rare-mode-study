"""Coverage metric v2 -- size-matched, threshold-free.

Why v1 was wrong
  v1 asked "is a generated sample within tau of each real sample", with tau a
  single global threshold.  That penalises classes whose samples are spread far
  apart (class 8, within-class spread 34.3) and rewards tight ones (class 1,
  spread 16.1).  It also ignored that the model emits ~5000 points while the
  real data has 10000, so the nearest-generated distance is inflated for every
  class.

v2 definition
  Split the real data into a reference half R and a query half Q.
  Generate |R| samples G from the server GMM + server decoder.
  For each class c and each query point q of class c:
      d_real(q) = distance from q to the nearest point of R
      d_gen(q)  = distance from q to the nearest point of G
  ratio(c) = median_q d_gen(q) / median_q d_real(q)

  ratio = 1  -> the generated set is as informative as an equal-sized real set
  ratio > 1  -> the model is worse than real data for that class

  No threshold is involved and R, G have the same size, so the comparison is
  matched in both sample count and class geometry.

Self-check
  Running with --self-check replaces G by R itself and must give ratio == 1.0
  for every class.  The script refuses to report otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def min_dist_to_set(A, B, chunk=1024):
    """distance from each row of A to the nearest row of B (Gram, low memory)."""
    A = np.ascontiguousarray(A, np.float32)
    B = np.ascontiguousarray(B, np.float32)
    sqB = (B * B).sum(1)
    out = np.empty(len(A))
    for i in range(0, len(A), chunk):
        j = min(len(A), i + chunk)
        a = A[i:j]
        d = (a * a).sum(1)[:, None] + sqB[None, :] - 2.0 * (a @ B.T)
        out[i:j] = np.sqrt(np.maximum(d.min(1), 0.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--mode", choices=("clusterlet", "author"), required=True)
    ap.add_argument("--n-gen", type=int, default=5000)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    run = args.run.resolve()
    sys.path.insert(0, str(args.source.resolve()))
    sys.path.insert(0, str(args.data_dir.resolve()))
    import fm_data
    from model import VaDE
    from types import SimpleNamespace

    cfg = json.loads((run / "config.json").read_text())
    hid = int(cfg.get("hid_dim", cfg.get("latent_dim", 10)))
    k, loaders, (X, y), sizes, _ = fm_data.make_clients(
        args.dataset, args.mode, cfg["clients"], cfg["batch_size"],
        data_dir=args.data_dir.resolve(), source_copy=run / "source", seed=42)
    Xn = X.numpy()
    Y = y.numpy()
    n = len(Xn)

    # deterministic, class-stratified half/half split
    rng = np.random.RandomState(args.split_seed)
    ref_idx, qry_idx = [], []
    for c in np.unique(Y):
        idx = np.where(Y == c)[0]
        rng.shuffle(idx)
        h = len(idx) // 2
        ref_idx.append(idx[:h])
        qry_idx.append(idx[h:])
    ref_idx = np.sort(np.concatenate(ref_idx))
    qry_idx = np.sort(np.concatenate(qry_idx))
    R = Xn[ref_idx]
    Q = Xn[qry_idx]
    Yq = Y[qry_idx]

    if args.self_check:
        G = R
    else:
        state = torch.load(run / "last_checkpoint.pt", map_location="cpu",
                           weights_only=False)["server"]
        model = VaDE(SimpleNamespace(input_dim=Xn.shape[1], hid_dim=hid,
                                     nClusters=len(state["rho_"]),
                                     alpha=float(cfg["alpha"]), cuda=False))
        model.load_state_dict(state, strict=True)
        model.eval()
        torch.manual_seed(0)
        np.random.seed(0)
        pi = torch.softmax(model.rho_, 0).detach().numpy()
        mu = model.mu_c.detach().numpy()
        var = torch.exp(model.log_sigma2_c).detach().numpy()
        comp = np.random.choice(len(pi), size=len(R), p=pi)
        z = mu[comp] + np.sqrt(var[comp]) * np.random.randn(len(R), mu.shape[1])
        with torch.no_grad():
            G = model.decoder(torch.from_numpy(z).float()).numpy()

    d_real = min_dist_to_set(Q, R)
    d_gen = min_dist_to_set(Q, G)

    rows = []
    for c in sorted(np.unique(Y)):
        sel = Yq == c
        if sel.sum() < 2:
            continue
        mr = float(np.median(d_real[sel]))
        mg = float(np.median(d_gen[sel]))
        rows.append({"class": int(c), "n_query": int(sel.sum()),
                     "median_d_real": round(mr, 4),
                     "median_d_gen": round(mg, 4),
                     "ratio": round(mg / mr, 4) if mr > 0 else None})
    overall = float(np.median(d_gen) / np.median(d_real))

    out = {"run": str(run), "dataset": args.dataset, "mode": args.mode,
           "hid_dim": hid, "n_ref": int(len(R)), "n_gen": int(len(G)),
           "self_check": bool(args.self_check),
           "overall_ratio": round(overall, 4),
           "per_class": rows}
    (run / ("coverage_v2_selfcheck.json" if args.self_check else "coverage_v2.json")
     ).write_text(json.dumps(out, indent=2) + "\n")

    print("=" * 74)
    print("%s  hid=%d  |R|=%d |G|=%d  %s" % (run.name, hid, len(R), len(G),
                                             "SELF-CHECK" if args.self_check else ""))
    print("%-6s %8s %12s %12s %10s" % ("class", "n_query", "median_d_real", "median_d_gen", "ratio"))
    for r in rows:
        print("%-6d %8d %12.3f %12.3f %10.3f"
              % (r["class"], r["n_query"], r["median_d_real"], r["median_d_gen"], r["ratio"]))
    print("%-6s %8s %12.3f %12.3f %10.3f" % ("ALL", len(Q), float(np.median(d_real)),
                                             float(np.median(d_gen)), overall))

    if args.self_check:
        bad = [r["class"] for r in rows if abs(r["ratio"] - 1.0) > 1e-3]
        if bad or abs(overall - 1.0) > 1e-3:
            print("\nSELF-CHECK FAILED for classes %s (overall %.6f)" % (bad, overall))
            sys.exit(1)
        print("\nSELF-CHECK PASSED: ratio == 1.0 for every class")


if __name__ == "__main__":
    main()

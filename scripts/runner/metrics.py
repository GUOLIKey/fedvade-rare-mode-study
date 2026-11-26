"""Evaluate a saved FedVaDE server checkpoint on Fashion-MNIST.

Reports
  * Global ARI / Purity / NMI, averaged over N inference seeds (author protocol:
    freeze the model, vary the latent sample inside predict())
  * per-class generative coverage: draw samples from the server GMM, decode with
    the server decoder, and measure how close each real sample is to the
    generated set, scaled by the median real-to-real nearest-neighbour distance

Coverage is the metric that matters for the rare-mode question: ARI is dominated
by the head classes, so a model can keep ARI while never generating the tail.

Usage
  python metrics.py --run <run dir> --data-dir <.../data> --source <FedVaDE_source> \
      --dataset fmnist_lt10 --mode clusterlet --rounds 100 300 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def cluster_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    d = int(max(y_true.max(), y_pred.max())) + 1
    conf = np.zeros((d, d), np.int64)
    for a, b in zip(y_true, y_pred):
        conf[a, b] += 1
    r, c = linear_sum_assignment(-conf)
    return {"ari": float(adjusted_rand_score(y_true, y_pred)),
            "purity": float(np.max(conf, 0).sum() / len(y_true)),
            "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
            "acc": float(conf[r, c].sum() / len(y_true))}


def min_dist(A, B, chunk=256):
    out = np.empty(len(A))
    for i in range(0, len(A), chunk):
        d = ((A[i:i + chunk, None, :] - B[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = np.sqrt(d.min(1))
    return out


def min_dist_self(A, chunk=256):
    """Nearest neighbour inside A, self excluded."""
    out = np.empty(len(A))
    for i in range(0, len(A), chunk):
        j = min(len(A), i + chunk)
        d = ((A[i:j, None, :] - A[None, :, :]) ** 2).sum(-1)
        d[np.arange(j - i), i + np.arange(j - i)] = np.inf
        out[i:j] = np.sqrt(d.min(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--mode", choices=("clusterlet", "author"), required=True)
    ap.add_argument("--rounds", type=int, nargs="+", default=[100, 300])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-gen", type=int, default=5000)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    run = a.run.resolve()
    sys.path.insert(0, str(a.source.resolve()))
    sys.path.insert(0, str(a.data_dir.resolve()))
    import fm_data
    from model import VaDE
    from types import SimpleNamespace

    cfg = json.loads((run / "config.json").read_text())
    hid = int(cfg.get("hid_dim", cfg.get("latent_dim", 10)))
    k, loaders, (X, y), sizes, _ = fm_data.make_clients(
        a.dataset, a.mode, cfg["clients"], cfg["batch_size"],
        data_dir=a.data_dir.resolve(), source_copy=run / "source", seed=42)
    Xn = X.numpy()
    Y = y.numpy()
    counts = {int(c): int(n) for c, n in zip(*np.unique(Y, return_counts=True))}
    tau = float(np.median(min_dist_self(Xn)))

    device = torch.device(a.device if (a.device == "cuda" and torch.cuda.is_available()) else "cpu")
    out = {"run": str(run), "dataset": a.dataset, "mode": a.mode,
           "class_counts": counts, "tau": tau, "evaluated_rounds": {}}

    for rnd in a.rounds:
        ck = run / ("epoch_%03d_server.pt" % rnd)
        if not ck.exists():
            out["evaluated_rounds"][rnd] = {"error": "missing checkpoint"}
            continue
        state = torch.load(ck, map_location="cpu", weights_only=False)["server"]
        model = VaDE(SimpleNamespace(input_dim=int(Xn.shape[1]), hid_dim=hid,
                                     nClusters=len(state["rho_"]), alpha=float(cfg["alpha"]),
                                     cuda=False))
        model.load_state_dict(state, strict=True)
        model.eval()

        # --- clustering metrics, frozen model, varying inference seed
        draws = []
        for s in range(a.seeds):
            torch.manual_seed(s)
            preds, labels = [], []
            with torch.no_grad():
                for ld in loaders:
                    for xb, yb in ld:
                        preds.append(model.predict(xb))
                        labels.append(yb.numpy())
            draws.append(cluster_metrics(np.concatenate(labels), np.concatenate(preds)))
        ari = np.array([d["ari"] for d in draws])
        pur = np.array([d["purity"] for d in draws])
        nmi = np.array([d["nmi"] for d in draws])

        # --- generative coverage from the server GMM + server decoder
        torch.manual_seed(0)
        np.random.seed(0)
        pi = torch.softmax(model.rho_, 0).detach().numpy()
        mu = model.mu_c.detach().numpy()
        var = torch.exp(model.log_sigma2_c).detach().numpy()
        comp = np.random.choice(len(pi), size=a.n_gen, p=pi)
        z = mu[comp] + np.sqrt(var[comp]) * np.random.randn(a.n_gen, mu.shape[1])
        with torch.no_grad():
            G = model.decoder(torch.from_numpy(z).float()).numpy()
        dgen = min_dist(Xn, G)
        cov = {}
        for c in sorted(counts):
            sel = Y == c
            cov[c] = {"n": counts[c], "share": float(sel.mean()),
                      "coverage": float((dgen[sel] <= tau).mean()),
                      "median_dist_over_tau": float(np.median(dgen[sel]) / tau)}
        overall = float((dgen <= tau).mean())

        out["evaluated_rounds"][rnd] = {
            "ari_mean": float(ari.mean()), "ari_sd": float(ari.std()),
            "purity_mean": float(pur.mean()), "purity_sd": float(pur.std()),
            "nmi_mean": float(nmi.mean()), "nmi_sd": float(nmi.std()),
            "coverage_overall": overall, "coverage_by_class": cov,
            "server_K": int(len(state["rho_"])),
        }
        print("round %3d  ARI %.4f±%.4f  Purity %.4f  coverage %.3f  K=%d"
              % (rnd, ari.mean(), ari.std(), pur.mean(), overall, len(state["rho_"])))

    (run / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()

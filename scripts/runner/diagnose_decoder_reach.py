"""Decoder check: can the server decoder render classes 5/8/9 at all?

This separates two very different failures that the coverage metric conflates:

  A. the decoder never learned to render footwear   -> fixing the GMM cannot help
  B. the decoder is fine but we never sample the right z -> the GMM is the target

Method
  * reconstruction error per class: x -> encoder.mu -> decoder, compare with x
  * "oracle coverage": decode the latents of REAL samples of class c and ask
    whether those reconstructions land within tau of the real class-c samples.
    This bypasses gmm_sampling entirely.
  * compare against the coverage already obtained from GMM-sampled z.

Read-only, no training.

Usage
  python diagnose_decoder.py --run runs/bal_clusterlet/s0 --data-dir data \
      --source <FedVaDE_source> --dataset fmnist_bal --mode clusterlet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def min_dist(A, B, chunk=1024):
    """Nearest distance from each row of A to any row of B, via Gram (no 3-D temp)."""
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


def min_dist_self(A, chunk=1024):
    A = np.ascontiguousarray(A, np.float32)
    sq = (A * A).sum(1)
    out = np.empty(len(A))
    for i in range(0, len(A), chunk):
        j = min(len(A), i + chunk)
        d = sq[i:j, None] + sq[None, :] - 2.0 * (A[i:j] @ A.T)
        d[np.arange(j - i), i + np.arange(j - i)] = np.inf
        out[i:j] = np.sqrt(np.maximum(d.min(1), 0.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--mode", choices=("clusterlet", "author"), required=True)
    ap.add_argument("--target-classes", type=int, nargs="+", default=[5, 8, 9])
    args = ap.parse_args()

    run = args.run.resolve()
    sys.path.insert(0, str(args.source.resolve()))
    sys.path.insert(0, str(args.data_dir.resolve()))
    import data_client_builder
    from model import VaDE
    from types import SimpleNamespace

    cfg = json.loads((run / "config.json").read_text())
    HID = int(cfg.get("hid_dim", cfg.get("latent_dim", 10)))
    k, loaders, (X, y), sizes, _ = data_client_builder.make_clients(
        args.dataset, args.mode, cfg["clients"], cfg["batch_size"],
        data_dir=args.data_dir.resolve(), source_copy=run / "source", seed=42)
    Xn = X.numpy()
    Y = y.numpy()
    counts = {int(c): int(n) for c, n in zip(*np.unique(Y, return_counts=True))}
    tau = float(np.median(min_dist_self(Xn)))
    T = args.target_classes

    ck_full = torch.load(run / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    state = ck_full["server"]
    model = VaDE(SimpleNamespace(input_dim=Xn.shape[1], hid_dim=HID,
                                 nClusters=len(state["rho_"]), alpha=float(cfg["alpha"]), cuda=False))
    model.load_state_dict(state, strict=True)
    model.eval()

    # client decoder / encoder (all clients hold identical weights after broadcast)
    c_state = ck_full["clients"][0]
    cm = VaDE(SimpleNamespace(input_dim=Xn.shape[1], hid_dim=HID,
                              nClusters=len(c_state["rho_"]), alpha=float(cfg["alpha"]), cuda=False))
    cm.load_state_dict(c_state, strict=True)
    cm.eval()

    # A. reconstruction path: x -> mu -> decoder
    with torch.no_grad():
        mu_all, _ = model.encoder(X)
        rec_all = model.decoder(mu_all).numpy()
        mu_cli, _ = cm.encoder(X)
        rec_cli = cm.decoder(mu_cli).numpy()

    # B. coverage from real latents (oracle), and from GMM sampling (reference)
    with torch.no_grad():
        torch.manual_seed(0); np.random.seed(0)
        pi = torch.softmax(model.rho_, 0).detach().numpy()
        mu_c = model.mu_c.detach().numpy()
        var_c = torch.exp(model.log_sigma2_c).detach().numpy()
        comp = np.random.choice(len(pi), size=5000, p=pi)
        z = mu_c[comp] + np.sqrt(var_c[comp]) * np.random.randn(5000, mu_c.shape[1])
        G = model.decoder(torch.from_numpy(z).float()).numpy()

    d_sampled = min_dist(Xn, G)
    # control: reconstruct every sample as its own class mean
    mean_rec = np.empty_like(Xn)
    for c in np.unique(Y):
        sel = Y == c
        mean_rec[sel] = Xn[sel].mean(0)

    rows = []
    for c in sorted(counts):
        sel = Y == c
        mse = float(((rec_all[sel] - Xn[sel]) ** 2).sum(1).mean())
        # oracle coverage: reconstructions OF class c, against real class c
        rec_c = rec_all[sel]
        dc = min_dist(Xn[sel], rec_c)
        rec_cc = rec_cli[sel]
        dcc = min_dist(Xn[sel], rec_cc)
        mse_cli = float(((rec_cli[sel] - Xn[sel]) ** 2).sum(1).mean())
        mse_mean = float(((mean_rec[sel] - Xn[sel]) ** 2).sum(1).mean())
        rows.append({"class": c, "n": counts[c],
                     "server_recon_mse": round(mse, 4),
                     "client_recon_mse": round(mse_cli, 4),
                     "classmean_recon_mse": round(mse_mean, 4),
                     "server_oracle_cov": round(float((dc <= tau).mean()), 4),
                     "client_oracle_cov": round(float((dcc <= tau).mean()), 4),
                     "sampled_coverage": round(float((d_sampled[sel] <= tau).mean()), 4),
                     "is_target": c in T})

    out = {"run": str(run), "tau": round(tau, 4), "classes": rows}
    (run / "decoder_check.json").write_text(json.dumps(out, indent=2) + "\n")

    print("=" * 78)
    print("%s  (%s / %s)" % (run.name, args.dataset, args.mode))
    print("tau = %.4f\n" % tau)
    print("%-6s %7s %9s %9s %9s %10s %10s %10s" % (
        "class", "n", "srv_mse", "cli_mse", "meanbase", "srv_orac", "cli_orac", "sampled"))
    for r in rows:
        print("%-6d %7d %9.1f %9.1f %9.1f %10.3f %10.3f %10.3f%s"
              % (r["class"], r["n"], r["server_recon_mse"], r["client_recon_mse"],
                 r["classmean_recon_mse"], r["server_oracle_cov"], r["client_oracle_cov"],
                 r["sampled_coverage"], "   <== 目标类" if r["is_target"] else ""))
    print()
    tgt = [r for r in rows if r["is_target"]]
    oth = [r for r in rows if not r["is_target"]]
    print("目标类  服务器oracle %.3f  客户端oracle %.3f  服务器mse %.0f  客户端mse %.0f"
          % (np.mean([r["server_oracle_cov"] for r in tgt]),
             np.mean([r["client_oracle_cov"] for r in tgt]),
             np.mean([r["server_recon_mse"] for r in tgt]),
             np.mean([r["client_recon_mse"] for r in tgt])))
    print("其余类  服务器oracle %.3f  客户端oracle %.3f  服务器mse %.0f  客户端mse %.0f"
          % (np.mean([r["server_oracle_cov"] for r in oth]),
             np.mean([r["client_oracle_cov"] for r in oth]),
             np.mean([r["server_recon_mse"] for r in oth]),
             np.mean([r["client_recon_mse"] for r in oth])))
    print()
    print("判断：oracle_cov 高而 sampled_cov 低 -> 解码器没问题，断点在 GMM 的分量/采样")
    print("      oracle_cov 也低              -> 解码器渲染不出该区域，改 GMM 也没用")


if __name__ == "__main__":
    main()

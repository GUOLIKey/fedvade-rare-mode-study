"""历史假设的诊断；旧指标不再支持“某些类生成不出来”的判断。

两个子命令，对应两次诊断：

  mode-loss   服务器分量构成 / 客户端局部 GMM / 潜空间可分性 / 每分量期望采样数
  decoder     解码器能否渲染某类：重构误差、oracle 覆盖率、服务器与客户端解码器对比

两次诊断都由 common/ 的共享工具完成计算，本文件只负责组织输出。

用法
  python lines/diagnostics/main.py mode-loss --run runs/平衡版_碎片化划分/s0 --data-dir data
  python lines/diagnostics/main.py decoder  --run runs/平衡版_碎片化划分/s0 --data-dir data
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import data as D            # noqa: E402
from common import distances as DIST    # noqa: E402
from common import model as M           # noqa: E402
from common import sampling as S        # noqa: E402


def _load(run: Path, data_dir: Path):
    cfg = json.loads((run / "config.json").read_text())
    state, payload, _ = M.load_server_state(run)
    k, loaders, (X, y), sizes, _ = D.make_clients(
        cfg["dataset"], cfg["partition_mode"], cfg["clients"], cfg["batch_size"],
        data_dir, run / "source", seed=42)
    model = M.build_vade(state, M.hid_dim_of(cfg), cfg["alpha"], X.shape[1], source=run / "source")
    return cfg, payload, state, model, k, loaders, X.numpy(), y.numpy()


# --------------------------------------------------------------------------- #
def cmd_mode_loss(a):
    cfg, payload, state, model, k, loaders, X, y = _load(a.run, a.data_dir)
    n_cls = int(len(np.unique(y)))
    counts = {int(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))}
    T = a.target_classes

    mu_all, assign = M.responsibilities(model, X)
    rows = M.component_table(state, assign, y, n_cls)
    K = len(state["rho_"])

    print("=" * 78)
    print("%s  (%s / %s)  round %s  N=%d  K_server=%d  sample=%d"
          % (a.run.name, cfg["dataset"], cfg["partition_mode"], payload.get("round"),
             len(y), K, cfg["sample"]))
    print("\n[1] 服务器 GMM")
    print("    死分量 %d / %d   最大分量占比 %.3f   每轮每分量期望样本 %.2f"
          % (sum(1 for r in rows if r["n"] == 0), K,
             max(r["share_of_data"] for r in rows), cfg["sample"] / K))
    for r in rows:
        print("    comp %2d  pi=%.3f  n=%5d  占比=%.3f  top=%s"
              % (r["component"], r["pi"], r["n"], r["share_of_data"], r.get("top_classes", [])))

    print("\n    目标类别落在哪些服务器分量:")
    placement = {}
    for c in T:
        sel = y == c
        if not sel.any():
            continue
        bc = np.bincount(assign[sel], minlength=K)
        placement[int(c)] = {int(i): int(v) for i, v in enumerate(bc) if v}
        print("      class %d (n=%d): %s" % (c, int(sel.sum()), placement[int(c)]))

    print("\n[2] 客户端局部 GMM")
    for cid, ld in enumerate(loaders):
        pairs = list(ld)
        xb = torch.cat([x for x, _ in pairs], 0)
        yb = torch.cat([t for _, t in pairs], 0).numpy()
        held = {int(t): int((yb == t).sum()) for t in T if (yb == t).sum() > 0}
        if not held:
            continue
        cm = M.build_vade(_client_state(a.run, cid), M.hid_dim_of(cfg), cfg["alpha"], xb.shape[1], source=a.run / "source")
        _, ca = M.responsibilities(cm, xb)
        tops = {}
        for j in range(len(cm.rho_)):
            sel = ca == j
            if sel.sum():
                tops[j] = int(np.bincount(yb[sel], minlength=n_cls).argmax())
        hit = {t: any(v == t for v in tops.values()) for t in held}
        print("    client %2d (n=%4d, K=%d) 持有 %s -> 有该类主导的分量（不保证纯簇） %s"
              % (cid, len(yb), len(cm.rho_), held, hit))

    print("\n[3] 潜空间可分性（1-近邻纯度）")
    overall, per, nn = DIST.knn_purity(mu_all, y)
    print("    整体 = %.4f" % overall)
    print("    %-6s %8s %8s %10s" % ("class", "n", "share", "purity"))
    for c, v in per.items():
        print("    %-6d %8d %7.1f%% %10.3f%s"
              % (c, v["n"], 100 * v["share"], v["purity"], "  <== 目标" if c in T else ""))
    print("\n    目标类的最近邻类别分布:")
    for c in T:
        sel = y == c
        if sel.any():
            print("      class %d -> %s" % (c, dict(Counter(y[nn[sel]].tolist()).most_common(4))))

    out = {"run": str(a.run), "server": rows, "placement": placement,
           "latent": {int(c): {"share": v["share"], "purity": v["purity"]} for c, v in per.items()},
           "counts": counts}
    from common.provenance import write_new_json
    write_new_json(a.output, out)


def _client_state(run: Path, cid: int):
    payload = torch.load(run / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    return payload["clients"][cid]


# --------------------------------------------------------------------------- #
def cmd_decoder(a):
    cfg, payload, state, model, k, loaders, X, y = _load(a.run, a.data_dir)
    T = a.target_classes
    counts = {int(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))}
    tau = float(np.median(DIST.min_dist_self(X)))

    # autoencoder path: x -> mu -> decoder, for the server and for one client
    mu_srv = M.encode_mean(model, X)
    rec_srv = S.decode(model, mu_srv)
    cli = M.build_vade(_client_state(a.run, 0), M.hid_dim_of(cfg), cfg["alpha"], X.shape[1], source=a.run / "source")
    rec_cli = S.decode(cli, M.encode_mean(cli, X))

    # generation path: GMM sampling (this is what actually feeds evaluation)
    G = S.generate(model, a.n_gen, seed=0)
    d_sampled = DIST.min_dist_to_set(X, G)

    # control: reconstruct as the class mean
    mean_rec = np.empty_like(X)
    for c in np.unique(y):
        sel = y == c
        mean_rec[sel] = X[sel].mean(0)

    rows = []
    for c in sorted(counts):
        sel = y == c
        mse_s = float(((rec_srv[sel] - X[sel]) ** 2).sum(1).mean())
        mse_c = float(((rec_cli[sel] - X[sel]) ** 2).sum(1).mean())
        mse_m = float(((mean_rec[sel] - X[sel]) ** 2).sum(1).mean())
        orc_s = float((DIST.min_dist_to_set(X[sel], rec_srv[sel]) <= tau).mean())
        orc_c = float((DIST.min_dist_to_set(X[sel], rec_cli[sel]) <= tau).mean())
        rows.append({"class": c, "n": counts[c], "server_recon_mse": round(mse_s, 4),
                     "client_recon_mse": round(mse_c, 4), "classmean_recon_mse": round(mse_m, 4),
                     "server_oracle_cov": round(orc_s, 4), "client_oracle_cov": round(orc_c, 4),
                     "sampled_coverage_global_tau": round(float((d_sampled[sel] <= tau).mean()), 4),
                     "is_target": c in T})

    print("=" * 78)
    print("%s  (%s / %s)  tau=%.4f" % (a.run.name, cfg["dataset"], cfg["partition_mode"], tau))
    print("%-6s %7s %9s %9s %9s %10s %10s"
          % ("class", "n", "srv_mse", "cli_mse", "meanbase", "srv_orac", "cli_orac"))
    for r in rows:
        print("%-6d %7d %9.1f %9.1f %9.1f %10.3f %10.3f%s"
              % (r["class"], r["n"], r["server_recon_mse"], r["client_recon_mse"],
                 r["classmean_recon_mse"], r["server_oracle_cov"], r["client_oracle_cov"],
                 "   <== 目标" if r["is_target"] else ""))
    tgt = [r for r in rows if r["is_target"]]
    oth = [r for r in rows if not r["is_target"]]
    if tgt:
        print("\n目标类 服务器oracle %.3f 客户端oracle %.3f"
              % (np.mean([r["server_oracle_cov"] for r in tgt]),
                 np.mean([r["client_oracle_cov"] for r in tgt])))
    if oth:
        print("其余类 服务器oracle %.3f 客户端oracle %.3f"
              % (np.mean([r["server_oracle_cov"] for r in oth]),
                 np.mean([r["client_oracle_cov"] for r in oth])))
    print("\n历史全局tau指标：仅保存诊断，不能据此定位跨类能力瓶颈")

    from common.provenance import write_new_json
    write_new_json(a.output, {"tau": tau, "classes": rows, "status": "historical_global_tau_diagnostic_only; do not infer class capability"})


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("mode-loss", cmd_mode_loss), ("decoder", cmd_decoder)):
        sp = sub.add_parser(name)
        sp.add_argument("--run", type=Path, required=True)
        sp.add_argument("--output", type=Path, required=True)
        sp.add_argument("--data-dir", type=Path, default=Path("data"))
        sp.add_argument("--target-classes", type=int, nargs="+", default=[5, 8, 9])
        if name == "decoder":
            sp.add_argument("--n-gen", type=int, default=5000)
        sp.set_defaults(func=fn)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()

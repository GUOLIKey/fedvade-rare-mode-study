"""Locate where classes 5/8/9 are lost: latent space, local GMMs, or server aggregation.

Four checks per checkpoint
  1. server GMM      : pi, dead components, dominant class per component, and
                       where the samples of classes 5/8/9 actually land
  2. client GMMs     : did clients holding 5/8/9 form components for them, and
                       did the variance degenerate
  3. latent space    : per-class 1-NN purity inside the encoder's mu, plus
                       raw-space comparison -- this is the control that decides
                       whether the loss is upstream of aggregation
  4. sampling budget : expected samples per server component per round

Everything is read-only.  No training.

Usage
  python diagnose_loss.py --run runs/bal_clusterlet/s0 --data-dir data \
      --source <FedVaDE_source> --dataset fmnist_bal --mode clusterlet
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def knn_purity(X, y, chunk=2000):
    X = np.ascontiguousarray(X, dtype=np.float32)
    sq = (X * X).sum(1)
    n = len(X)
    nn = np.empty(n, dtype=np.int32)
    for i in range(0, n, chunk):
        j = min(n, i + chunk)
        d = sq[i:j, None] + sq[None, :] - 2.0 * (X[i:j] @ X.T)
        d[np.arange(j - i), i + np.arange(j - i)] = np.inf
        nn[i:j] = d.argmin(1)
    per = {}
    for c in np.unique(y):
        sel = y == c
        per[int(c)] = {"n": int(sel.sum()), "share": float(sel.mean()),
                       "purity": float((y[nn[sel]] == c).mean())}
    return float((y[nn] == y).mean()), per, nn


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
    import fm_data
    from model import VaDE
    from types import SimpleNamespace

    cfg = json.loads((run / "config.json").read_text())
    HID = int(cfg.get("hid_dim", cfg.get("latent_dim", 10)))
    k, loaders, (X, y), sizes, _ = fm_data.make_clients(
        args.dataset, args.mode, cfg["clients"], cfg["batch_size"],
        data_dir=args.data_dir.resolve(), source_copy=run / "source", seed=42)
    Xall = X.numpy()
    Yall = y.numpy()
    n_cls = int(len(np.unique(Yall)))
    counts = {int(c): int(n) for c, n in zip(*np.unique(Yall, return_counts=True))}
    T = args.target_classes

    ck = torch.load(run / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    sv, clients = ck["server"], ck["clients"]

    def build(state):
        m = VaDE(SimpleNamespace(input_dim=Xall.shape[1], hid_dim=HID,
                                 nClusters=len(state["rho_"]), alpha=float(cfg["alpha"]), cuda=False))
        m.load_state_dict(state, strict=True)
        m.eval()
        return m

    def resp(model, xb):
        with torch.no_grad():
            mu, _ = model.encoder(xb)
            lp = model.gaussian_pdfs_log(mu, model.mu_c, model.log_sigma2_c) \
                + torch.log(torch.softmax(model.rho_, 0)).unsqueeze(0)
        return mu.numpy(), lp.argmax(1).numpy()

    out = {"run": str(run), "dataset": args.dataset, "mode": args.mode,
           "round": int(ck.get("round", -1)), "class_counts": counts,
           "target_classes": T}

    # ---------- 1. server GMM ----------
    sm = build(sv)
    mu_all, asg = resp(sm, X)
    K = int(len(sv["rho_"]))
    pi = torch.softmax(sm.rho_, 0).detach().numpy()
    cnt = np.bincount(asg, minlength=K)
    comp = []
    for c in range(K):
        sel = asg == c
        n = int(sel.sum())
        row = {"component": c, "pi": round(float(pi[c]), 4), "n": n,
               "share_of_data": round(n / len(Yall), 4),
               "mean_log_sigma2": round(float(sm.log_sigma2_c[c].mean()), 3)}
        if n:
            bc = np.bincount(Yall[sel], minlength=n_cls)
            top = bc.argsort()[::-1][:3]
            row["top_classes"] = [[int(t), int(bc[t])] for t in top if bc[t] > 0]
        comp.append(row)
    out["server"] = {
        "K": K, "dead_components": int((cnt == 0).sum()),
        "max_component_share": round(float(cnt.max() / len(Yall)), 4),
        "components": comp,
        "expected_samples_per_component_per_round": round(float(cfg["sample"]) / K, 2),
    }
    # where do target classes land?
    tgt = {}
    for c in T:
        sel = Yall == c
        if not sel.any():
            continue
        bc = np.bincount(asg[sel], minlength=K)
        tgt[int(c)] = {"n": int(sel.sum()),
                       "server_components_used": {int(i): int(v) for i, v in enumerate(bc) if v},
                       "n_components_used": int((bc > 0).sum())}
    out["target_class_server_placement"] = tgt

    # ---------- 2. client local GMMs ----------
    cli = []
    for cid, state in enumerate(clients):
        xb_all = torch.cat([xb for xb, _ in loaders[cid]], 0)
        yb_all = torch.cat([yb for _, yb in loaders[cid]], 0).numpy()
        cm = build(state)
        Kl = int(len(state["rho_"]))
        _, ca = resp(cm, xb_all)
        p = torch.softmax(cm.rho_, 0).detach().numpy()
        rows = []
        for j in range(Kl):
            sel = ca == j
            n = int(sel.sum())
            r = {"comp": j, "n": n, "pi": round(float(p[j]), 4),
                 "mean_log_sigma2": round(float(cm.log_sigma2_c[j].mean()), 3)}
            if n:
                bc = np.bincount(yb_all[sel], minlength=n_cls)
                r["top_class"] = int(bc.argmax())
                r["top_class_n"] = int(bc.max())
                r["target_hits"] = {int(t): int(bc[t]) for t in T if t < len(bc) and bc[t] > 0}
            rows.append(r)
        holds = {int(t): int((yb_all == t).sum()) for t in T if (yb_all == t).sum() > 0}
        has_target_comp = {int(t): any(r.get("top_class") == t for r in rows) for t in holds}
        cli.append({"client": cid, "n": int(len(yb_all)), "local_K": Kl,
                    "holds_target": holds, "target_has_component": has_target_comp,
                    "components": rows})
    out["clients"] = cli

    # ---------- 3. latent separability (the control) ----------
    lat_overall, lat_per, lat_nn = knn_purity(mu_all, Yall)
    raw_overall, raw_per, _ = knn_purity(Xall, Yall)
    out["latent"] = {"overall_1nn": round(lat_overall, 4), "per_class": {
        int(c): {"n": v["n"], "share": round(v["share"], 4), "purity": round(v["purity"], 4)}
        for c, v in lat_per.items()}}
    out["raw"] = {"overall_1nn": round(raw_overall, 4), "per_class": {
        int(c): {"n": v["n"], "share": round(v["share"], 4), "purity": round(v["purity"], 4)}
        for c, v in raw_per.items()}}
    # for each target class, what is its nearest-neighbour class in latent space?
    conf = {}
    for c in T:
        sel = Yall == c
        if not sel.any():
            continue
        conf[int(c)] = {int(k): int(v) for k, v in
                        Counter(Yall[lat_nn[sel]].tolist()).most_common(4)}
    out["latent_nearest_class_of_targets"] = conf

    (run / "diagnosis.json").write_text(json.dumps(out, indent=2) + "\n")

    # ---------- print ----------
    print("=" * 78)
    print("%s  (%s / %s)  round %d  N=%d  K_server=%d  sample=%d"
          % (run.name, args.dataset, args.mode, out["round"], len(Yall), K, cfg["sample"]))
    print("\n[1] 服务器 GMM")
    print("    死分量 %d / %d   最大分量占比 %.3f   每轮每分量期望样本 %.2f"
          % (out["server"]["dead_components"], K, out["server"]["max_component_share"],
             out["server"]["expected_samples_per_component_per_round"]))
    for r in comp:
        print("    comp %2d  pi=%.3f  n=%5d  占比=%.3f  top=%s"
              % (r["component"], r["pi"], r["n"], r["share_of_data"], r.get("top_classes", [])))
    print("\n    目标类别落在哪些服务器分量:")
    for c, v in tgt.items():
        print("      class %d (n=%d): %s" % (c, v["n"], v["server_components_used"]))

    print("\n[2] 客户端局部 GMM")
    for r in cli:
        if not r["holds_target"]:
            continue
        print("    client %2d (n=%4d, K=%d) 持有 %s" % (r["client"], r["n"], r["local_K"], r["holds_target"]))
        for t, ok in r["target_has_component"].items():
            hits = [(x["comp"], x["n"], x["pi"], x["mean_log_sigma2"])
                    for x in r["components"] if x.get("top_class") == t]
            print("        class %d: 有自己的分量=%s  %s" % (t, ok, hits[:3]))

    print("\n[3] 潜空间 vs 原始空间（1-近邻纯度）")
    print("    %-6s %8s %8s %10s %10s" % ("class", "n", "share", "raw", "latent"))
    for c in sorted(raw_per):
        print("    %-6d %8d %7.1f%% %10.3f %10.3f%s"
              % (c, raw_per[c]["n"], 100 * raw_per[c]["share"],
                 raw_per[c]["purity"], lat_per[c]["purity"],
                 "   <== 目标类" if c in T else ""))
    print("    整体: raw %.4f / latent %.4f" % (raw_overall, lat_overall))
    print("\n    目标类的最近邻类别分布（潜空间）:")
    for c, v in conf.items():
        print("      class %d -> %s" % (c, v))

    # ---------- verdict ----------
    print("\n[4] 判断依据")
    lat_ok = all(lat_per[c]["purity"] > 3 * lat_per[c]["share"] for c in T if c in lat_per)
    srv_missing = all(t.get("n_components_used", 0) <= 1 for t in tgt.values()) if tgt else False
    print("    潜空间里目标类可分? %s" % ("是" if lat_ok else "否"))
    print("    目标类只用到一个服务器分量? %s" % ("是" if srv_missing else "否"))
    print("    -> 若 latent 可分但目标类挤在同一分量，断点在服务器聚合")
    print("    -> 若 latent 不可分，断点在编码器，改服务器无效")


if __name__ == "__main__":
    main()

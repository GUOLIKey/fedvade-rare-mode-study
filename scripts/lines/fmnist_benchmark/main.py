"""线路：碎片化基准（Fashion-MNIST + Clusterlets 划分）

一条线路一个文件。包含五件事，用子命令区分：

  prepare   下载 Fashion-MNIST，构造平衡版 / 长尾版，缓存
  check     训练前的数据可用性检验（每类 1-近邻纯度、集中式 GMM、划分统计）
  run       跑一次完整联邦训练（委托给 common/train.py）
  driver    三组划分 × 3 个种子，训练并评估
  eval      对已有 run 计算 Global ARI / Purity / NMI（10 个推理种子）

用法
  python lines/fmnist_benchmark/main.py prepare --data-dir data
  python lines/fmnist_benchmark/main.py check   --data-dir data --dataset fmnist_bal
  python lines/fmnist_benchmark/main.py driver  --source ../FedVaDE_source --data-dir data
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import data as D            # noqa: E402
from common import distances as DIST    # noqa: E402
from common import metrics as MET       # noqa: E402
from common import model as M           # noqa: E402
from common import train as T           # noqa: E402
from common.evaluate import evaluate_run  # noqa: E402

DATASETS = {"fmnist_bal": None, "fmnist_lt10": 10.0, "fmnist_lt50": 50.0}

# 三组：名字 -> (数据集, 划分方式)
GROUPS = [
    ("平衡版·碎片化划分", "fmnist_bal", "clusterlet"),
    ("长尾版·碎片化划分", "fmnist_lt10", "clusterlet"),
    ("长尾版·作者划分",   "fmnist_lt10", "author"),
]
SEEDS = [0, 1, 2]


# --------------------------------------------------------------------------- #
def cmd_prepare(a):
    for name, rho in DATASETS.items():
        path = D.build_dataset(a.data_dir, name, rho, n_target=a.n)
        X, y = D.load_dataset(a.data_dir, name)
        counts = {int(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))}
        print("%-12s N=%d  counts=%s" % (name, len(y), counts))


def cmd_check(a):
    X, y = D.load_dataset(a.data_dir, a.dataset)
    Xs, _ = D.standardize(X)
    k = int(len(np.unique(y)))
    print("=" * 74)
    print("%s  N=%d  k=%d  dim=%d" % (a.dataset, len(y), k, Xs.shape[1]))

    overall, per, _ = DIST.knn_purity(Xs, y)
    print("\n[1] 1-近邻同类比例  整体 = %.4f" % overall)
    print("    %-6s %8s %8s %10s" % ("class", "n", "share", "purity"))
    for c, v in per.items():
        print("    %-6d %8d %7.1f%% %10.3f" % (c, v["n"], 100 * v["share"], v["purity"]))

    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    print("\n[2] 集中式 KMeans ARI = %.4f" % adjusted_rand_score(y, km.labels_))
    gm = GaussianMixture(n_components=k, covariance_type="diag", random_state=42,
                         reg_covar=1e-3, n_init=1, max_iter=100).fit(Xs)
    print("[3] 集中式 GMM    ARI = %.4f" % adjusted_rand_score(y, gm.predict(Xs)))

    parts, stats = D.make_clusterlet_partition(Xs, y, n_clients=a.clients)
    print("\n[4] 碎片化划分：重叠样本 %d（必须为 0），每客户端类数 %s"
          % (stats["overlap_samples"], stats["classes_per_client"]))
    print("    每类被几个客户端持有:", stats["clients_per_class"])


def cmd_run(a):
    from types import SimpleNamespace
    args = T.build_parser().parse_args([
        "--source", str(a.source), "--out", str(a.out), "--data-dir", str(a.data_dir),
        "--dataset", a.dataset, "--mode", a.mode, "--seed", str(a.seed),
        "--rounds", str(a.rounds), "--hid-dim", str(a.hid_dim), "--device", a.device])
    T.run(args)


def cmd_driver(a):
    out_root = a.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    for label, dataset, mode in GROUPS:
        for seed in SEEDS:
            out = out_root / label.replace("·", "_") / ("s%d" % seed)
            if not out.exists():
                print(json.dumps({"group": label, "seed": seed, "status": "running"}), flush=True)
                args = T.build_parser().parse_args([
                    "--source", str(a.source), "--out", str(out), "--data-dir", str(a.data_dir),
                    "--dataset", dataset, "--mode", mode, "--seed", str(seed),
                    "--rounds", str(a.rounds), "--device", a.device])
                T.run(args)
            evaluate(out, dataset, mode, a.seeds_inf, a.data_dir)
    summarize(out_root)


def cmdeval_eval(a):
    dataset, mode = a.dataset, a.mode
    if dataset is None or mode is None:
        cfg = json.loads((a.run / "config.json").read_text())
        dataset, mode = cfg["dataset"], cfg["partition_mode"]
    evaluate(a.run, dataset, mode, a.seeds_inf, a.data_dir, a.output)


def evaluate(run: Path, dataset, mode, n_inference, data_dir, output=None):
    out = evaluate_run(run, dataset, mode, n_inference, data_dir=data_dir, output=output)
    if "300" in out["rounds"]:
        print("%-28s %-4s ARI@300 = %.4f" % (run.parent.name, run.name,
              out["rounds"]["300"]["ari"]["mean"]))


def summarize(out_root: Path):
    print("\n" + "=" * 78)
    print("%-22s %-4s %10s %10s" % ("group", "seed", "ARI@100", "ARI@300"))
    agg = {}
    for label, _, _ in GROUPS:
        key = label.replace("·", "_")
        for seed in SEEDS:
            f = out_root / key / ("s%d" % seed) / "metrics.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())["rounds"]
            a100 = r.get("100", {}).get("ari", {}).get("mean")
            a300 = r.get("300", {}).get("ari", {}).get("mean")
            print("%-22s %-4d %10s %10s" % (label, seed,
                  "%.4f" % a100 if a100 else "-", "%.4f" % a300 if a300 else "-"))
            if a300:
                agg.setdefault(label, []).append(a300)
    print("\n%-22s %20s" % ("group", "ARI@300"))
    for label, v in agg.items():
        print("%-22s %10.4f ± %.4f" % (label, statistics.mean(v),
                                       statistics.stdev(v) if len(v) > 1 else 0))


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare"); sp.add_argument("--data-dir", type=Path, default=Path("data"))
    sp.add_argument("--n", type=int, default=10000); sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("check")
    sp.add_argument("--data-dir", type=Path, default=Path("data"))
    sp.add_argument("--dataset", default="fmnist_bal"); sp.add_argument("--clients", type=int, default=10)
    sp.set_defaults(func=cmd_check)

    sp = sub.add_parser("run")
    sp.add_argument("--source", type=Path, required=True); sp.add_argument("--out", type=Path, required=True)
    sp.add_argument("--data-dir", type=Path, default=Path("data"))
    sp.add_argument("--dataset", default="fmnist_bal"); sp.add_argument("--mode", default="clusterlet")
    sp.add_argument("--seed", type=int, default=0); sp.add_argument("--rounds", type=int, default=300)
    sp.add_argument("--hid-dim", type=int, default=10); sp.add_argument("--device", default="cuda")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("driver")
    sp.add_argument("--source", type=Path, required=True)
    sp.add_argument("--data-dir", type=Path, default=Path("data"))
    sp.add_argument("--out-root", type=Path, default=Path("runs"))
    sp.add_argument("--rounds", type=int, default=300)
    sp.add_argument("--seeds-inf", type=int, default=10)
    sp.add_argument("--device", default="cuda"); sp.set_defaults(func=cmd_driver)

    sp = sub.add_parser("eval")
    sp.add_argument("--run", type=Path, required=True)
    sp.add_argument("--output", type=Path, required=True)
    sp.add_argument("--dataset", default=None); sp.add_argument("--mode", default=None)
    sp.add_argument("--data-dir", type=Path)
    sp.add_argument("--seeds-inf", type=int, default=10); sp.set_defaults(func=cmdeval_eval)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()

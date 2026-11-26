"""线路：客户端 DPGMM（用 Dirichlet 过程替代"用标签定 K"）

四个子命令：

  dump      跑 50 轮预训练，导出每个客户端的 local_mu，然后停下
  precheck  在导出的 local_mu 上离线拟合 DPGMM，统计各 prior 保留多少分量
  driver    对选定的 prior 跑完整 300 轮（可选同 K 普通 GMM 对照），并评估
  eval      对已有 run 计算 Global ARI / Purity / NMI

方法本身（DPGMM 拟合 + 剪枝 + 同 K 对照）在 common/local_gmm.py；
训练循环在 common/train.py，本文件只做组织。

用法
  python lines/client_dpgmm/main.py dump     --source ../FedVaDE_source --data-dir data
  python lines/client_dpgmm/main.py precheck --dump dump_bal
  python lines/client_dpgmm/main.py driver   --source ../FedVaDE_source --data-dir data --wcp 0.1
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
from common import local_gmm as LG      # noqa: E402
from common import metrics as MET       # noqa: E402
from common import model as M           # noqa: E402
from common import train as T           # noqa: E402
from common.evaluate import evaluate_run  # noqa: E402

DATASETS = {"fmnist_bal": None, "fmnist_lt10": 10.0}


def _train_args(a, out, dataset, local_gmm, dump=None):
    argv = ["--source", str(a.source), "--out", str(out), "--data-dir", str(a.data_dir),
            "--dataset", dataset, "--mode", "clusterlet", "--local-gmm", local_gmm,
            "--wcp", str(a.wcp), "--seed", str(a.seed), "--rounds", str(a.rounds),
            "--device", a.device]
    if dump:
        argv += ["--dump-pretrain", str(dump)]
    return T.build_parser().parse_args(argv)


# --------------------------------------------------------------------------- #
def cmd_dump(a):
    for dataset in DATASETS:
        dump = a.out_dir / dataset
        print("===== dump %s =====" % dataset, flush=True)
        try:
            T.run(_train_args(a, a.out_dir / ("_tmp_" + dataset), dataset, "author", dump=dump))
        except SystemExit:
            pass


def cmd_precheck(a):
    files = sorted(a.dump.glob("client_*_mu.npy"))
    if not files:
        raise SystemExit("no client_*_mu.npy in %s" % a.dump)
    mus = [np.load(f) for f in files]
    print("=" * 74)
    print("%s  clients=%d  每客户端样本数=%s"
          % (a.dump.name, len(mus), [len(m) for m in mus]))
    for wcp in a.priors:
        ks = []
        for m in mus:
            w, _, _ = LG.fit_dpgmm(m, wcp, n_components=a.n_components, prune=a.prune)
            ks.append(len(w))
        total = int(sum(ks))
        print("\nDP 先验 = %-6s 每客户端保留 K = %s" % (wcp, ks))
        print("  总分量数 = %2d   每轮每分量期望样本 = %.2f"
              % (total, a.sample / total if total else 0))
    print("\n判据：总分量数 <= 60 可用；> 80 视为分量爆炸（服务器每轮只有 sample 个采样名额）")


def cmd_driver(a):
    out_root = a.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    for dataset in a.datasets:
        for mode in a.local_modes:
            out = out_root / ("%s_%s" % (dataset, mode)) / ("s%d" % a.seed)
            if out.exists():
                continue
            print(json.dumps({"dataset": dataset, "local_gmm": mode, "seed": a.seed}), flush=True)
            T.run(_train_args(a, out, dataset, mode))
            evaluate_run(out, dataset, 'clusterlet', data_dir=a.data_dir)


def cmdeval_eval(a):
    cfg = json.loads((a.run / "config.json").read_text())
    evaluate_run(a.run, cfg["dataset"], 'clusterlet', a.seeds_inf, data_dir=a.data_dir, output=a.output)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("dump")
    sp.add_argument("--source", type=Path, required=True)
    sp.add_argument("--data-dir", type=Path, default=Path("data"))
    sp.add_argument("--out-dir", type=Path, default=Path("dump"))
    sp.add_argument("--rounds", type=int, default=1); sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--wcp", type=float, default=0.1); sp.add_argument("--device", default="cuda")
    sp.set_defaults(func=cmd_dump)

    sp = sub.add_parser("precheck")
    sp.add_argument("--dump", type=Path, required=True)
    sp.add_argument("--priors", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    sp.add_argument("--prune", type=float, default=0.01)
    sp.add_argument("--n-components", type=int, default=15)
    sp.add_argument("--sample", type=int, default=100)
    sp.set_defaults(func=cmd_precheck)

    sp = sub.add_parser("driver")
    sp.add_argument("--source", type=Path, required=True)
    sp.add_argument("--data-dir", type=Path, default=Path("data"))
    sp.add_argument("--out-root", type=Path, default=Path("runs_dpgmm"))
    sp.add_argument("--datasets", nargs="+", default=["fmnist_bal", "fmnist_lt10"])
    sp.add_argument("--local-modes", nargs="+", default=["dpgmm", "samek_gmm"])
    sp.add_argument("--wcp", type=float, default=0.1)
    sp.add_argument("--seed", type=int, default=0); sp.add_argument("--rounds", type=int, default=300)
    sp.add_argument("--device", default="cuda")
    sp.set_defaults(func=cmd_driver)

    sp = sub.add_parser("eval")
    sp.add_argument("--run", type=Path, required=True)
    sp.add_argument("--output", type=Path, required=True)
    sp.add_argument("--data-dir", type=Path)
    sp.add_argument("--seeds-inf", type=int, default=10); sp.set_defaults(func=cmdeval_eval)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()

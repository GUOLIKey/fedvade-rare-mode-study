"""线路：潜空间宽度（hid_dim 扫描）

假说：生成覆盖受潜空间宽度限制。三组宽度 10 / 20 / 30，各 3 个训练种子。
10 维那组复用碎片化基准线的结果，本文件只跑 20 / 30。

用法
  python lines/latent_width/main.py --source ../FedVaDE_source --data-dir data
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import train as T                       # noqa: E402
from common.evaluate import evaluate_run            # noqa: E402

WIDTHS = [20, 30]
SEEDS = [0, 1, 2]


def train_args(a, out, width, seed):
    return T.build_parser().parse_args([
        "--source", str(a.source), "--out", str(out), "--data-dir", str(a.data_dir),
        "--dataset", a.dataset, "--mode", a.mode, "--seed", str(seed),
        "--rounds", str(a.rounds), "--hid-dim", str(width), "--device", a.device])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-root", type=Path, default=Path("runs_latent_width"))
    ap.add_argument("--dataset", default="fmnist_bal")
    ap.add_argument("--mode", default="clusterlet")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--widths", nargs="*", type=int, default=WIDTHS)
    ap.add_argument("--baseline-root", type=Path, default=Path("runs"),
                    help="hid_dim=10 results from the benchmark line, if available")
    a = ap.parse_args()

    a.out_root.mkdir(parents=True, exist_ok=True)
    for width in a.widths:
        for seed in SEEDS:
            out = a.out_root / ("hid%d" % width) / ("s%d" % seed)
            if not out.exists():
                print(json.dumps({"hid_dim": width, "seed": seed}), flush=True)
                T.run(train_args(a, out, width, seed))
            evaluate_run(out, a.dataset, a.mode)

    print("\n" + "=" * 66)
    print("%-10s %6s %20s" % ("hid_dim", "n", "ARI@300"))
    for width, base in ([(10, a.baseline_root)] + [(w, a.out_root / ("hid%d" % w)) for w in a.widths]):
        vals = []
        for seed in SEEDS:
            f = base / ("s%d" % seed) / "metrics.json"
            if f.exists():
                m = json.loads(f.read_text())["rounds"].get("300", {}).get("ari", {}).get("mean")
                if m is not None:
                    vals.append(m)
        if vals:
            print("%-10s %6d %10.4f ± %.4f" % ("hid=%d" % width, len(vals),
                  statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0))


if __name__ == "__main__":
    main()

"""线路：覆盖率指标（size-matched，无阈值）

旧指标的缺陷与修正过程见 docs/reports/16_correction.md。本文件实现修好之后的版本。

定义
  真实数据按类分层对半分：参考集 R、查询集 Q
  模型生成 |R| 条样本 G
  ratio(c) = median_{q∈Q_c} d(q, G) / median_{q∈Q_c} d(q, R)

  ratio = 1 -> 生成集和同等大小的真实集一样有用
  ratio > 1 -> 模型在该类上更差

  参考集与生成集大小相同、查询点相同、类内几何自动抵消，不需要任何阈值。

自检
  --self-check 把生成集换成参考集本身，每个类的 ratio 必须精确等于 1.0；
  不通过则拒绝输出结果。

用法
  python lines/coverage_metric/main.py --run runs/平衡版_碎片化划分/s0 --data-dir data
  python lines/coverage_metric/main.py --run runs/平衡版_碎片化划分/s0 --data-dir data --self-check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import data as D            # noqa: E402
from common import distances as DIST    # noqa: E402
from common import model as M           # noqa: E402
from common import sampling as S        # noqa: E402


def compute(run: Path, data_dir: Path, n_gen: int | None, split_seed: int, self_check: bool):
    cfg = json.loads((run / "config.json").read_text())
    dataset, mode = cfg["dataset"], cfg["partition_mode"]
    k, loaders, (X, y), sizes, _ = D.make_clients(
        dataset, mode, cfg["clients"], cfg["batch_size"], data_dir, run / "source")
    Xn, Y = X.numpy(), y.numpy()

    ref_idx, qry_idx = D.half_split(Y, seed=split_seed)
    R, Q, Yq = Xn[ref_idx], Xn[qry_idx], Y[qry_idx]

    if self_check:
        G = R
    else:
        state, _, _ = M.load_server_state(run)
        model = M.build_vade(state, M.hid_dim_of(cfg), cfg["alpha"], Xn.shape[1])
        G = S.generate(model, n_gen or len(R), seed=0)

    d_real = DIST.min_dist_to_set(Q, R)
    d_gen = DIST.min_dist_to_set(Q, G)

    rows = []
    for c in sorted(np.unique(Y)):
        sel = Yq == c
        if sel.sum() < 2:
            continue
        mr, mg = float(np.median(d_real[sel])), float(np.median(d_gen[sel]))
        rows.append({"class": int(c), "n_query": int(sel.sum()),
                     "median_d_real": round(mr, 4), "median_d_gen": round(mg, 4),
                     "ratio": round(mg / mr, 4) if mr > 0 else None})
    overall = float(np.median(d_gen) / np.median(d_real))
    return {"run": str(run), "dataset": dataset, "mode": mode,
            "n_ref": int(len(R)), "n_gen": int(len(G)), "self_check": bool(self_check),
            "overall_ratio": round(overall, 4), "per_class": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--n-gen", type=int, default=None, help="default: same size as the reference set")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()

    res = compute(a.run, a.data_dir, a.n_gen, a.split_seed, a.self_check)

    print("=" * 74)
    print("%s  |R|=%d |G|=%d  %s"
          % (a.run.name, res["n_ref"], res["n_gen"], "SELF-CHECK" if a.self_check else ""))
    print("%-6s %8s %12s %12s %10s"
          % ("class", "n_query", "median_d_real", "median_d_gen", "ratio"))
    for r in res["per_class"]:
        print("%-6d %8d %12.3f %12.3f %10.3f"
              % (r["class"], r["n_query"], r["median_d_real"], r["median_d_gen"], r["ratio"]))
    print("%-6s %8s %12s %12s %10.3f" % ("ALL", "", "", "", res["overall_ratio"]))

    if a.self_check:
        bad = [r["class"] for r in res["per_class"] if abs(r["ratio"] - 1.0) > 1e-3]
        if bad or abs(res["overall_ratio"] - 1.0) > 1e-3:
            print("\nSELF-CHECK FAILED: %s (overall %.6f)" % (bad, res["overall_ratio"]))
            raise SystemExit(1)
        print("\nSELF-CHECK PASSED: ratio == 1.0 for every class")
        (a.run / "coverage_v2_selfcheck.json").write_text(json.dumps(res, indent=2) + "\n")
    else:
        (a.run / "coverage_v2.json").write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()

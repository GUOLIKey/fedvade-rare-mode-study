"""Baseline driver: original FedVaDE on Fashion-MNIST under three settings.

Groups
  bal_clusterlet : balanced Fashion-MNIST, Clusterlets partition
  lt10_clusterlet: long-tailed (rho=10),      Clusterlets partition
  lt10_author    : long-tailed (rho=10),      the author's own partition (control)

Each group runs seeds 0/1/2.  Clustering metrics and generative coverage are
evaluated at rounds 100 and 300 with 10 inference seeds.

Usage
  python driver_baseline.py --phase smoke --device cuda     # 2 rounds, wiring check
  python driver_baseline.py --phase full  --device cuda     # 9 runs + evaluation
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
DATA = EXP / "data"
RUNNER = HERE / "runner_cl_vade.py"
METRICS = HERE / "metrics.py"

GROUPS = [
    ("bal_clusterlet", "fmnist_bal", "clusterlet"),
    ("lt10_clusterlet", "fmnist_lt10", "clusterlet"),
    ("lt10_author", "fmnist_lt10", "author"),
]
SEEDS = [0, 1, 2]
BASE = dict(lr="0.002", alpha="0.05", schedule="author", sample="100",
            pretrain="50", batch_size="200", clients="10")


def run_one(tag, dataset, mode, seed, source, device, phase, out_root):
    out = out_root / tag / ("s%d" % seed)
    rounds = 2 if phase == "smoke" else 300
    save_every = 1 if phase == "smoke" else 25
    if out.exists():
        print("skip existing", out, flush=True)
    else:
        cmd = [sys.executable, str(RUNNER), "--source", str(source), "--out", str(out),
               "--data-dir", str(DATA), "--dataset", dataset, "--mode", mode,
               "--seed", str(seed), "--device", device,
               "--lr", BASE["lr"], "--alpha", BASE["alpha"],
               "--schedule", BASE["schedule"], "--sample", BASE["sample"],
               "--pretrain", BASE["pretrain"], "--batch-size", BASE["batch_size"],
               "--clients", BASE["clients"], "--rounds", str(rounds),
               "--save-every", str(save_every)]
        t0 = time.time()
        out.parent.mkdir(parents=True, exist_ok=True)
        with (out.parent / ("s%d.log" % seed)).open("w") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
        print(json.dumps({"tag": tag, "seed": seed, "rc": rc,
                          "seconds": round(time.time() - t0)}), flush=True)
        if rc != 0:
            return {"tag": tag, "seed": seed, "rc": rc}
    if phase == "smoke":
        return {"tag": tag, "seed": seed, "rc": 0, "smoke": True}
    cmd = [sys.executable, str(METRICS), "--run", str(out), "--data-dir", str(DATA),
           "--source", str(source), "--dataset", dataset, "--mode", mode,
           "--rounds", "100", "300", "--device", "cpu"]
    with (out.parent / ("s%d.metrics.log" % seed)).open("w") as fh:
        mrc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return {"tag": tag, "seed": seed, "rc": 0, "metrics_rc": mrc}


def summarize(out_root, seeds=None):
    print("\n" + "=" * 96)
    print("%-16s %-4s %10s %10s %10s %10s %10s" % (
        "group", "seed", "ARI@100", "ARI@300", "cov@100", "cov@300", "tail_cov@300"))
    table = {}
    for tag, dataset, mode in GROUPS:
        rows = []
        for seed in seeds:
            mf = out_root / tag / ("s%d" % seed) / "metrics.json"
            if not mf.exists():
                continue
            m = json.loads(mf.read_text())
            r = m["evaluated_rounds"]
            cell = {"ari100": r.get("100", {}).get("ari_mean"),
                    "ari300": r.get("300", {}).get("ari_mean"),
                    "cov100": r.get("100", {}).get("coverage_overall"),
                    "cov300": r.get("300", {}).get("coverage_overall"),
                    "tail300": (r.get("300", {}).get("coverage_by_class", {})
                                .get("9", {}).get("coverage"))}
            rows.append(cell)
            print("%-16s %-4d %10s %10s %10s %10s %10s" % (
                tag, seed,
                "%.4f" % cell["ari100"] if cell["ari100"] is not None else "-",
                "%.4f" % cell["ari300"] if cell["ari300"] is not None else "-",
                "%.3f" % cell["cov100"] if cell["cov100"] is not None else "-",
                "%.3f" % cell["cov300"] if cell["cov300"] is not None else "-",
                "%.3f" % cell["tail300"] if cell["tail300"] is not None else "-"))
        if rows:
            for key in ("ari100", "ari300", "cov100", "cov300"):
                vals = [r[key] for r in rows if r[key] is not None]
                if vals:
                    table.setdefault(tag, {})[key] = {
                        "mean": round(statistics.mean(vals), 4),
                        "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else None}
    print("\n%-16s %10s %10s %10s %10s" % ("group", "ARI@100", "ARI@300", "cov@100", "cov@300"))
    for tag in table:
        c = table[tag]
        def f(k):
            v = c.get(k)
            if not v:
                return "-"
            return "%.4f±%.3f" % (v["mean"], v["sd"] or 0)
        print("%-16s %10s %10s %10s %10s" % (tag, f("ari100"), f("ari300"), f("cov100"), f("cov300")))
    (EXP / "baseline_summary.json").write_text(json.dumps(table, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=EXP.parent.parent / "FedVaDE_source")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--phase", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    a = ap.parse_args()
    out_root = EXP / ("runs_smoke" if a.phase == "smoke" else "runs")
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = a.seeds if a.seeds else SEEDS
    status = []
    for tag, dataset, mode in GROUPS:
        if a.groups and tag not in a.groups:
            continue
        for seed in seeds:
            status.append(run_one(tag, dataset, mode, seed, a.source.resolve(),
                                  a.device, a.phase, out_root))
            (EXP / "driver_status.json").write_text(json.dumps(status, indent=2) + "\n")
    if a.phase == "full":
        summarize(out_root, seeds)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()

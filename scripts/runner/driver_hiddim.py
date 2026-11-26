"""hid_dim sweep on balanced Fashion-MNIST, Clusterlets partition.

Tests the hypothesis that generative coverage is bounded by latent width,
not by federated aggregation.  hid_dim=10 is the existing bal_clusterlet run.
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
OUT = EXP / "runs_hiddim"
SEEDS = [0, 1, 2]
HID = [20, 30]
BASE = dict(lr="0.002", alpha="0.05", schedule="author", sample="100",
            pretrain="50", batch_size="200", clients="10")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=EXP.parent.parent / "FedVaDE_source")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dataset", default="fmnist_bal")
    ap.add_argument("--mode", default="clusterlet")
    ap.add_argument("--hid-dims", nargs="*", type=int, default=HID)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--rounds", type=int, default=300)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    src = a.source.resolve()
    for hd in a.hid_dims:
        for seed in a.seeds:
            out = OUT / ("hid%d" % hd) / ("s%d" % seed)
            if out.exists():
                print("skip", out, flush=True)
            else:
                out.parent.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, str(RUNNER), "--source", str(src), "--out", str(out),
                       "--data-dir", str(DATA), "--dataset", a.dataset, "--mode", a.mode,
                       "--seed", str(seed), "--device", a.device, "--hid-dim", str(hd),
                       "--lr", BASE["lr"], "--alpha", BASE["alpha"],
                       "--schedule", BASE["schedule"], "--sample", BASE["sample"],
                       "--pretrain", BASE["pretrain"], "--batch-size", BASE["batch_size"],
                       "--clients", BASE["clients"], "--rounds", str(a.rounds),
                       "--save-every", "25"]
                t0 = time.time()
                with (out.parent / ("s%d.log" % seed)).open("w") as fh:
                    rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
                print(json.dumps({"hid": hd, "seed": seed, "rc": rc,
                                  "seconds": round(time.time() - t0)}), flush=True)
                if rc != 0:
                    continue
            cmd = [sys.executable, str(METRICS), "--run", str(out), "--data-dir", str(DATA),
                   "--source", str(src), "--dataset", a.dataset, "--mode", a.mode,
                   "--rounds", "100", "300", "--device", "cpu"]
            with (out.parent / ("s%d.metrics.log" % seed)).open("w") as fh:
                subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)

    # summary, with the existing hid=10 run as the reference
    print("\n" + "=" * 92)
    print("%-10s %6s %10s %10s %10s   per-class coverage @300 (5 / 8 / 9)")
    rows = {}
    ref = EXP / "runs" / "bal_clusterlet"
    for hd, base in ([(10, ref)] + [(h, OUT / ("hid%d" % h)) for h in a.hid_dims]):
        for seed in a.seeds:
            mf = base / ("s%d" % seed) / "metrics.json"
            if not mf.exists():
                continue
            m = json.loads(mf.read_text())
            r = m["evaluated_rounds"].get("300")
            if not r:
                continue
            cb = r["coverage_by_class"]
            rows.setdefault(hd, []).append({
                "ari": r["ari_mean"], "cov": r["coverage_overall"],
                "c5": cb.get("5", {}).get("coverage", float("nan")),
                "c8": cb.get("8", {}).get("coverage", float("nan")),
                "c9": cb.get("9", {}).get("coverage", float("nan"))})
    for hd in sorted(rows):
        v = rows[hd]
        f = lambda k: statistics.mean([x[k] for x in v])
        print("%-10s %6d %10.4f %10.3f %10.3f   %.3f / %.3f / %.3f"
              % ("hid=%d" % hd, len(v), f("ari"), f("cov"), f("cov"), f("c5"), f("c8"), f("c9")))
    (EXP / "hiddim_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()

"""Run one FedVaDE configuration on Fashion-MNIST under a chosen partition.

The author's four .py files are copied into <out>/source and patched there only.
FedVaDE_source/ is never modified.

Three injections into the copy's main.py
  1. data loading   -> _make_data(...)  (clusterlets or author partition)
  2. client losses  -> _client_loss(...) per client per round, for NaN triage
  3. round observer -> _observe(...)     (metrics, NaN scan, checkpoints)

Usage
  python runner_cl_vade.py --source <FedVaDE_source> --out <dir> \
      --data-dir <experiments/fmnist_clusterlets_v1/data> \
      --dataset fmnist_lt10 --mode clusterlet --seed 0 \
      --lr 0.002 --alpha 0.05 --schedule author --sample 100 \
      --rounds 300 --pretrain 50 --save-every 25 --device cuda
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import platform
import random
import shutil
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("expected exactly one match for %r (found %d)"
                         % (old[:60], source.count(old)))
    return source.replace(old, new)


def atomic_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def instrument(source: str, pretrain: int) -> str:
    # 0. pretraining length (author hardcodes pre_epoch=50 in the call)
    if pretrain != 50:
        source = once(source, "pre_epoch=50)", "pre_epoch=%d)" % pretrain)
    # 1. data loading
    source = once(
        source,
        "    args.nClusters, DL, (X, y), size = make_noniid_clients(csv_path, args.client_num, args.batch_size, args.dataset)",
        "    args.nClusters, DL, (X, y), size = _make_data(args.client_num, args.batch_size)")
    # 2. per-client loss, for attributing numerical failures to a client
    source = once(
        source,
        "                    L_c += loss.detach().cpu().numpy()",
        "                    L_c += loss.detach().cpu().numpy()\n"
        "                    _client_loss(c['client_id'], float(loss.detach().cpu()))")
    # 3. round observer
    point = "            Result.append([L_c, L_s, purity, ari])"
    source = once(source, point,
                  point + "\n            _observe(epoch + 1, server, clients, DL, L_c, L_s, purity, ari)")
    return source


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--mode", choices=("clusterlet", "author"), required=True)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--schedule", choices=("author", "epoch"), default="author")
    p.add_argument("--sample", type=int, default=100)
    p.add_argument("--rounds", type=int, default=300)
    p.add_argument("--pretrain", type=int, default=50)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--hid-dim", type=int, default=10)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = p.parse_args()

    src, out = args.source.resolve(), args.out.resolve()
    data_dir = args.data_dir.resolve()
    if out.exists():
        raise SystemExit("existing run directory: %s" % out)
    out.mkdir(parents=True)
    copy = out / "source"
    copy.mkdir()
    for path in src.glob("*.py"):
        shutil.copy2(path, copy / path.name)

    entry = copy / "main.py"
    original = entry.read_text()
    updated = instrument(original, args.pretrain)
    (out / "source.diff").write_text("".join(difflib.unified_diff(
        original.splitlines(True), updated.splitlines(True),
        "author/main.py", "copy/main.py")))
    entry.write_text(updated)

    config = {"dataset": args.dataset, "partition_mode": args.mode, "seed": args.seed,
              "lr": args.lr, "alpha": args.alpha, "schedule": args.schedule,
              "sample": args.sample, "pretrain": args.pretrain, "rounds": args.rounds,
              "save_every": args.save_every, "batch_size": args.batch_size,
              "clients": args.clients, "latent_dim": args.hid_dim, "device": args.device,
              "source_dir": str(src),
              "author_source_hashes": {q.name: sha(q) for q in src.glob("*.py")},
              "executed_main_sha": sha(entry),
              "command": [sys.executable, *sys.argv],
              "platform": platform.platform()}
    atomic_json(out / "config.json", config)

    sys.path.insert(0, str(copy))
    sys.path.insert(0, str(data_dir))
    sys.dont_write_bytecode = True
    import torch
    from threadpoolctl import threadpool_limits

    if args.device == "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    config["torch"] = torch.__version__
    config["numpy"] = np.__version__
    config["cuda_available"] = bool(torch.cuda.is_available())
    if args.device == "cuda" and torch.cuda.is_available():
        config["gpu"] = torch.cuda.get_device_name(0)
    atomic_json(out / "config.json", config)

    import data_client_builder
    parse_args = argparse.ArgumentParser.parse_args
    rows, client_losses = [], {}
    nan_events = []
    started = perf_counter()

    def _make_data(n_clients, batch_size):
        k, loaders, (X, y), sizes, stats = data_client_builder.make_clients(
            args.dataset, args.mode, n_clients, batch_size,
            data_dir=data_dir, source_copy=copy, seed=42)
        config["partition"] = stats
        config["input_dim"] = int(X.shape[1])
        config["clusters"] = int(k)
        config["client_sizes"] = sizes
        atomic_json(out / "config.json", config)
        return k, loaders, (X, y), sizes

    def _client_loss(cid, value):
        client_losses[int(cid)] = value

    def _scan(state):
        bad = []
        for key, tensor in state.items():
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                if not torch.isfinite(tensor).all():
                    bad.append(key)
        return bad

    def snapshot(epoch, server, clients, full=False):
        model_state = {k: v.detach().cpu().clone() for k, v in server["vade"].state_dict().items()}
        torch.save({"round": epoch, "server": model_state}, out / ("epoch_%03d_server.pt" % epoch))
        if full:
            torch.save({"round": epoch, "seed": args.seed, "server": model_state,
                        "clients": [{k: v.detach().cpu().clone() for k, v in c["vade"].state_dict().items()}
                                    for c in clients]}, out / "last_checkpoint.pt")

    def observe(epoch, server, clients, loaders, lc, ls, purity, ari):
        server_bad = _scan(server["vade"].state_dict())
        client_bad = {c["client_id"]: _scan(c["vade"].state_dict()) for c in clients}
        client_bad = {k: v for k, v in client_bad.items() if v}
        losses = {k: client_losses.get(k) for k in range(len(clients))}
        loss_bad = [k for k, v in losses.items() if v is not None and not np.isfinite(v)]
        row = {"round": epoch, "client_loss": float(lc), "server_loss": float(ls),
               "purity": float(purity), "ari": float(ari),
               "client_losses": losses,
               "server_lr": server["optimizer"].param_groups[0]["lr"],
               "nonfinite_server_params": server_bad,
               "nonfinite_client_params": client_bad,
               "nonfinite_client_losses": loss_bad,
               "elapsed_seconds": perf_counter() - started}
        rows.append(row)
        with (out / "rounds.jsonl").open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        if server_bad or client_bad or loss_bad:
            nan_events.append({"round": epoch, "server": server_bad,
                               "clients": client_bad, "client_losses": loss_bad})
            atomic_json(out / "nan_events.json", nan_events)
        client_losses.clear()
        if epoch % args.save_every == 0 or epoch == args.rounds:
            snapshot(epoch, server, clients, full=(epoch == args.rounds))

    def force_device(parser, *a, **kw):
        parsed = parse_args(parser, *a, **kw)
        if hasattr(parsed, "cuda"):
            parsed.cuda = (args.device == "cuda")
        return parsed

    author_argv = [str(entry), "--dataset", args.dataset, "--epoch", str(args.rounds),
                   "--lr", str(args.lr), "--alpha", str(args.alpha),
                   "--sample", str(args.sample), "--client_num", str(args.clients),
                   "--batch_size", str(args.batch_size), "--hid_dim", str(args.hid_dim)]
    result = {"status": "running", "config": config, "author_argv": author_argv}
    atomic_json(out / "result.json", result)
    try:
        with threadpool_limits(limits=1), \
                patch.object(sys, "argv", author_argv), \
                patch.object(argparse.ArgumentParser, "parse_args", force_device):
            namespace = {"__name__": "__main__", "__file__": str(entry),
                         "_observe": observe, "_client_loss": _client_loss,
                         "_make_data": _make_data}
            exec(compile(updated, str(entry), "exec"), namespace)
        if len(rows) != args.rounds:
            raise RuntimeError("incomplete rounds: %d / %d" % (len(rows), args.rounds))
        result.update(status="ok", last=rows[-1], nan_events=len(nan_events))
    except BaseException as exc:
        result.update(status="failed", error=repr(exc), nan_events=len(nan_events))
        raise
    finally:
        result["elapsed_seconds"] = perf_counter() - started
        atomic_json(out / "result.json", result)


if __name__ == "__main__":
    main()

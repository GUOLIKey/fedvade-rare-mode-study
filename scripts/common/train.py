"""The single federated training driver, shared by every experiment line.

Merges what used to be two runners (plain FedVaDE / DPGMM local GMM).  The
author's four .py files are copied into <out>/source and patched there only;
FedVaDE_source/ is never modified.

Three injections into the copy's main.py
  1. data loading   -> _make_data(...)
  2. client losses  -> _client_loss(...)  per client per round, for NaN triage
  3. round observer -> _observe(...)      metrics, NaN scan, checkpoints
and, optionally, one into the copy's model.py that replaces the local-GMM block.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import platform
import random
import shutil
import sys
import os
import subprocess
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'common'

from . import data as data_mod
from .provenance import dataset_file, sha256, array_sha256, source_hashes, script_fingerprint, runtime_versions


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("expected exactly one match for %r (found %d)"
                         % (old[:60], source.count(old)))
    return source.replace(old, new)


def _atomic_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


OLD_LOCAL_GMM = """            K_l = len(np.unique(local_y))
            gmm = GaussianMixture(n_components=K_l, covariance_type='diag', random_state=42)
            gmm.fit(local_mu)

            pi_ = torch.tensor(gmm.weights_, dtype=torch.float, device=device)
            c['vade'].rho_.data = torch.log(pi_ + epsilon)
            c['vade'].mu_c.data = torch.tensor(gmm.means_, dtype=torch.float, device=device)
            c['vade'].log_sigma2_c.data = torch.log(torch.tensor(gmm.covariances_, dtype=torch.float, device=device))
"""

NEW_LOCAL_GMM = """            _w, _m, _s, _k = _fit_local(local_mu, local_y)
            _record_k(c['client_id'], _k)
            pi_ = torch.tensor(_w, dtype=torch.float, device=device)
            c['vade'].rho_.data = torch.log(pi_ + epsilon)
            c['vade'].mu_c.data = torch.tensor(_m, dtype=torch.float, device=device)
            c['vade'].log_sigma2_c.data = torch.log(torch.tensor(_s, dtype=torch.float, device=device))
"""


def instrument_main(source: str, pretrain: int) -> str:
    source = _once(source,
                   "        pre_train(args, server, clients, weight, pre_epoch=50)\n",
                   "        pre_train(args, server, clients, weight, pre_epoch=50)\n"
                   "        _dump_pretrain(server, clients, DL)\n")
    if pretrain != 50:
        source = _once(source, "pre_epoch=50)", "pre_epoch=%d)" % pretrain)
    source = _once(source,
                   "    args.nClusters, DL, (X, y), size = make_noniid_clients(csv_path, args.client_num, args.batch_size, args.dataset)",
                   "    args.nClusters, DL, (X, y), size = _make_data(args.client_num, args.batch_size)")
    source = _once(source,
                   "                    L_c += loss.detach().cpu().numpy()",
                   "                    L_c += loss.detach().cpu().numpy()\n"
                   "                    _client_loss(c['client_id'], float(loss.detach().cpu()))")
    point = "            Result.append([L_c, L_s, purity, ari])"
    source = _once(source, point,
                   point + "\n            _observe(epoch + 1, server, clients, DL, L_c, L_s, purity, ari)")
    return source


def patch_schedule(source, schedule):
    if schedule == 'author':
        return source
    source = _once(source, "                    c['lr_s'].step()\n", '')
    return _once(source, "                clients_vae_param.append(c['vade'].state_dict())",
                 "                c['lr_s'].step()\n                clients_vae_param.append(c['vade'].state_dict())")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one FedVaDE configuration.")
    p.add_argument("--source", type=Path, required=True, help="path to FedVaDE_source")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--mode", choices=("clusterlet", "author"), default="clusterlet")
    p.add_argument("--local-gmm", choices=("author", "dpgmm", "samek_gmm"), default="author")
    p.add_argument("--wcp", type=float, default=0.1, help="DP weight_concentration_prior")
    p.add_argument("--prune", type=float, default=0.01)
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
    p.add_argument('--sample-floor', type=int, default=0)
    p.add_argument('--usage-floor', type=float, default=0.)
    p.add_argument('--usage-beta', type=float, default=4.)
    p.add_argument("--dump-pretrain", type=Path, default=None,
                   help="save each client's local_mu after pretraining and stop")
    return p


def run(args) -> None:
    """One clean interpreter per run; no author-module/cache leakage."""
    args.source, args.out, args.data_dir = args.source.resolve(), args.out.resolve(), args.data_dir.resolve()
    if args.dump_pretrain is not None:
        args.dump_pretrain = args.dump_pretrain.resolve()
    argv = []
    for key, value in vars(args).items():
        if value is not None:
            argv.extend(['--'+key.replace('_','-'), str(value)])
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get('PYTHONPATH', '')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    subprocess.run([sys.executable, '-m', 'common.train', *argv], env=env, check=True)


def _run_in_process(args) -> None:
    src, out = args.source.resolve(), args.out.resolve()
    data_dir = args.data_dir.resolve()
    if args.rounds < 1 or args.pretrain < 1 or args.sample < 1 or args.clients < 1:
        raise ValueError('rounds, pretrain, sample and clients must be positive')
    if args.sample_floor < 0 or not 0 <= args.usage_floor <= 1 or args.usage_beta < 0:
        raise ValueError('invalid sampling/loss floor parameters')
    if out.exists():
        raise SystemExit("existing run directory: %s" % out)
    out.mkdir(parents=True)
    copy = out / "source"
    copy.mkdir()
    for path in src.glob("*.py"):
        shutil.copy2(path, copy / path.name)

    entry = copy / "main.py"
    original = entry.read_text()
    updated = patch_schedule(instrument_main(original, args.pretrain), args.schedule)
    (out / "source.diff").write_text("".join(difflib.unified_diff(
        original.splitlines(True), updated.splitlines(True), "author/main.py", "copy/main.py")))
    entry.write_text(updated)

    if args.local_gmm != "author":
        model_py = copy / "model.py"
        s = model_py.read_text()
        if s.count(OLD_LOCAL_GMM) != 1:
            raise ValueError("local-GMM block not found exactly once in model.py")
        model_py.write_text(s.replace(OLD_LOCAL_GMM, NEW_LOCAL_GMM))

    if args.sample_floor or args.usage_floor:
        from lines.server_reweight.main import apply
        apply(copy/'model.py', args.sample_floor, args.usage_floor, args.usage_beta)

    for name in ('model.py',):
        old = (src/name).read_text()
        new = (copy/name).read_text()
        (out/(name+'.diff')).write_text(''.join(difflib.unified_diff(
            old.splitlines(True), new.splitlines(True), 'author/'+name, 'copy/'+name)))

    config = {"dataset": args.dataset, "partition_mode": args.mode, "seed": args.seed,
              "lr": args.lr, "alpha": args.alpha, "schedule": args.schedule,
              "sample": args.sample, "pretrain": args.pretrain, "rounds": args.rounds,
              "save_every": args.save_every, "batch_size": args.batch_size,
              "clients": args.clients, "latent_dim": args.hid_dim, "device": args.device,
              "local_gmm": args.local_gmm, "wcp": args.wcp, "prune": args.prune,
              "source_dir": str(src), "platform": platform.platform(),
              "author_source_hashes": {q.name: _sha(q) for q in src.glob("*.py")},
              "executed_main_sha": _sha(entry), 'executed_source_hashes': source_hashes(copy),
              'data_dir': str(data_dir), 'data_sha256': sha256(dataset_file(data_dir,args.dataset)),
              'command': [sys.executable, *sys.argv],
              'repair_package': script_fingerprint(Path(__file__).resolve().parents[1]),
              'runtime_packages': runtime_versions(), 'python':sys.version,
              'sample_floor': args.sample_floor, 'usage_floor': args.usage_floor, 'usage_beta': args.usage_beta}
    _atomic_json(out / "config.json", config)

    sys.path.insert(0, str(copy))
    sys.path.insert(0, str(data_dir))
    sys.dont_write_bytecode = True
    import torch
    from threadpoolctl import threadpool_limits
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    os.chdir(out)

    if args.device == "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    config.update(torch=torch.__version__, numpy=np.__version__,
                  cuda_available=bool(torch.cuda.is_available()))
    _atomic_json(out / "config.json", config)

    from . import local_gmm as lg
    parse_args = argparse.ArgumentParser.parse_args
    rows, client_losses, nan_events, retained_k = [], {}, [], {}
    started = perf_counter()

    def _make_data(n_clients, batch_size):
        k, loaders, (X, y), sizes, stats = data_mod.make_clients(
            args.dataset, args.mode, n_clients, batch_size, data_dir, copy, seed=42)
        config.update(partition=stats, input_dim=int(X.shape[1]), clusters=int(k),
                      client_sizes=sizes, input_sha256=array_sha256(X.numpy()),
                      label_sha256=array_sha256(y.numpy()))
        # Read TensorDataset tensors directly: iterating shuffled loaders would consume training RNG.
        config['client_data_hashes'] = [
            {'x': array_sha256(ld.dataset.tensors[0].numpy()),
             'y': array_sha256(ld.dataset.tensors[1].numpy())} for ld in loaders]
        if stats.get('input_csv'):
            config['input_csv_sha256'] = sha256(stats['input_csv'])
        _atomic_json(out / "config.json", config)
        return k, loaders, (X, y), sizes

    def _client_loss(cid, value):
        client_losses[int(cid)] = value

    def _fit_local(local_mu, local_y):
        if args.local_gmm == "author":
            from sklearn.mixture import GaussianMixture
            k = len(np.unique(local_y))
            g = GaussianMixture(n_components=k, covariance_type="diag", random_state=42).fit(local_mu)
            return g.weights_, g.means_, g.covariances_, k
        return lg.fit_local(local_mu, args.local_gmm, args.wcp, prune=args.prune)

    def _record_k(cid, k):
        retained_k[int(cid)] = int(k)
        config["retained_K"] = retained_k
        _atomic_json(out / "config.json", config)

    def _scan(state):
        return [key for key, t in state.items()
                if torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all()]

    def _dump_pretrain(server, clients, loaders):
        if args.dump_pretrain is None:
            return
        d = Path(args.dump_pretrain).resolve()
        d.mkdir(parents=True, exist_ok=True)
        dev = next(clients[0]["vade"].parameters()).device
        with torch.no_grad():
            for c, ld in zip(clients, loaders):
                mus = [c["vade"].encoder(xb.to(dev))[0].cpu() for xb, _ in ld]
                np.save(d / ("client_%02d_mu.npy" % c["client_id"]), torch.cat(mus, 0).numpy())
        (d / "meta.json").write_text(json.dumps(
            {"dataset": args.dataset, "mode": args.mode, "seed": args.seed,
             "hid_dim": args.hid_dim, "clients": args.clients}, indent=2) + "\n")
        print("pretrain dump written to %s" % d, flush=True)
        raise SystemExit(0)

    def _snapshot(epoch, server, clients, full=False):
        state = {k: v.detach().cpu().clone() for k, v in server["vade"].state_dict().items()}
        torch.save({"round": epoch, "server": state}, out / ("epoch_%03d_server.pt" % epoch))
        if full:
            torch.save({"round": epoch, "seed": args.seed, "server": state,
                        'capability': 'evaluation_only; optimizer/scheduler/RNG not saved',
                        "clients": [{k: v.detach().cpu().clone() for k, v in c["vade"].state_dict().items()}
                                    for c in clients]}, out / "last_checkpoint.pt")

    def _observe(epoch, server, clients, loaders, lc, ls, purity, ari):
        srv_bad = _scan(server["vade"].state_dict())
        cli_bad = {c["client_id"]: _scan(c["vade"].state_dict()) for c in clients}
        cli_bad = {k: v for k, v in cli_bad.items() if v}
        losses = {k: client_losses.get(k) for k in range(len(clients))}
        bad_loss = [k for k, v in losses.items() if v is not None and not np.isfinite(v)]
        row = {"round": epoch, "client_loss": float(lc), "server_loss": float(ls),
               "purity": float(purity), "ari": float(ari), "client_losses": losses,
               "server_lr": server["optimizer"].param_groups[0]["lr"],
               'client_lr': [c['optimizer'].param_groups[0]['lr'] for c in clients],
               "nonfinite_server_params": srv_bad, "nonfinite_client_params": cli_bad,
               "nonfinite_client_losses": bad_loss,
               "elapsed_seconds": perf_counter() - started}
        rows.append(row)
        with (out / "rounds.jsonl").open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        if srv_bad or cli_bad or bad_loss:
            nan_events.append({"round": epoch, "server": srv_bad, "clients": cli_bad,
                               "client_losses": bad_loss})
            _atomic_json(out / "nan_events.json", nan_events)
            raise FloatingPointError('nonfinite loss/parameters; see nan_events.json')
        client_losses.clear()
        if epoch % args.save_every == 0 or epoch == args.rounds:
            _snapshot(epoch, server, clients, full=(epoch == args.rounds))

    def _force_device(parser, *a, **kw):
        parsed = parse_args(parser, *a, **kw)
        if hasattr(parsed, "cuda"):
            parsed.cuda = (args.device == "cuda")
        return parsed

    argv = [str(entry), "--dataset", args.dataset, "--epoch", str(args.rounds),
            "--lr", str(args.lr), "--alpha", str(args.alpha), "--sample", str(args.sample),
            "--client_num", str(args.clients), "--batch_size", str(args.batch_size),
            "--hid_dim", str(args.hid_dim)]
    result = {"status": "running", "config": config, "author_argv": argv}
    _atomic_json(out / "result.json", result)
    try:
        import model as _model_mod
        _model_mod._fit_local = _fit_local
        _model_mod._record_k = _record_k
        config['imported_model_file'] = str(Path(_model_mod.__file__).resolve())
        if Path(_model_mod.__file__).resolve() != (copy/'model.py').resolve():
            raise RuntimeError('wrong author model imported')
        _atomic_json(out/'config.json',config)
        with threadpool_limits(limits=1), \
                patch.object(sys, "argv", argv), \
                patch.object(argparse.ArgumentParser, "parse_args", _force_device):
            namespace = {"__name__": "__main__", "__file__": str(entry),
                         "_observe": _observe, "_client_loss": _client_loss,
                         "_make_data": _make_data, "_dump_pretrain": _dump_pretrain}
            exec(compile(updated, str(entry), "exec"), namespace)
        if len(rows) != args.rounds:
            raise RuntimeError("incomplete rounds: %d / %d" % (len(rows), args.rounds))
        result.update(status="ok", last=rows[-1], nan_events=len(nan_events))
    except SystemExit as exc:
        result.update(status='dump-only' if args.dump_pretrain is not None and exc.code in (0,None)
                      else 'failed', exit_code=exc.code)
        raise
    except BaseException as exc:
        result.update(status="failed", error=repr(exc), nan_events=len(nan_events))
        raise
    finally:
        result["elapsed_seconds"] = perf_counter() - started
        _atomic_json(out / "result.json", result)


def main():
    _run_in_process(build_parser().parse_args())


if __name__ == "__main__":
    main()

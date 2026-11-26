"""Fashion-MNIST loading, the Clusterlets partition, and client construction.

Merges what used to be three files (fmnist prep / partition / client builder)
because they are always used together and every training line needs them.
"""
from __future__ import annotations

import gzip
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

CACHE_NAME = "cache"


# --------------------------------------------------------------------------- #
# raw data
# --------------------------------------------------------------------------- #
def _read_idx(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        ndim = magic & 0xFF
        dims = [int.from_bytes(f.read(4), "big") for _ in range(ndim)]
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return buf.reshape(dims)


def load_fmnist_raw(raw_dir) -> tuple[np.ndarray, np.ndarray]:
    """Flattened [0,1] train split.  Pure numpy, no torchvision at load time
    (the local torch/numpy/python combination segfaults in its loader)."""
    raw_dir = Path(raw_dir)
    base = raw_dir / "FashionMNIST" / "raw"
    img, lab = base / "train-images-idx3-ubyte", base / "train-labels-idx1-ubyte"
    if not img.exists() or not lab.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        from torchvision import datasets       # download only
        datasets.FashionMNIST(root=str(raw_dir), train=True, download=True)
    X = _read_idx(img).reshape(-1, 784).astype(np.float32) / 255.0
    return X, _read_idx(lab).astype(np.int64)


def stratified_subsample(X, y, n_target, seed=0):
    rng = np.random.RandomState(seed)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        m = min(len(idx), max(1, int(round(n_target * len(idx) / len(y)))))
        keep.append(rng.choice(idx, m, replace=False))
    sel = np.sort(np.concatenate(keep))
    return X[sel], y[sel]


def make_long_tailed(X, y, rho, n_target=10000, seed=0):
    """Class i keeps round(n_max * rho^(-i/(k-1))) samples, i by descending frequency."""
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    counts = np.array([int((y == c).sum()) for c in classes])
    order = classes[np.argsort(-counts)]
    head = int(counts.max())
    keep = {}
    for rank, c in enumerate(order):
        target = max(1, int(round(head * (1.0 / rho) ** (rank / max(1, len(order) - 1)))))
        idx = np.where(y == c)[0]
        if target < len(idx):
            idx = rng.choice(idx, target, replace=False)
        keep[int(c)] = idx
    sel = np.sort(np.concatenate(list(keep.values())))
    Xs, ys = X[sel], y[sel]
    if len(ys) > n_target:
        Xs, ys = stratified_subsample(Xs, ys, n_target, seed=seed)
    return Xs, ys


def standardize(X, ref=None):
    sc = StandardScaler().fit(X)
    out = sc.transform(X).astype(np.float32)
    return out if ref is None else (out, sc.transform(ref).astype(np.float32))


def build_dataset(data_dir, name, rho, n_target=10000, seed=0) -> Path:
    """Cache a prepared dataset under data_dir/cache/<name>.npz (raw, unstandardised)."""
    cache = Path(data_dir) / CACHE_NAME
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / (name + ".npz")
    if out.exists():
        return out
    X, y = load_fmnist_raw(Path(data_dir) / "raw")
    Xs, ys = (stratified_subsample(X, y, n_target, seed) if rho is None
              else make_long_tailed(X, y, rho, n_target, seed))
    counts = {int(c): int(n) for c, n in zip(*np.unique(ys, return_counts=True))}
    meta = {"name": name, "rho": rho, "n": int(len(ys)), "class_counts": counts}
    np.savez_compressed(out, X=Xs, y=ys, meta=np.array(str(meta)))
    return out


def load_dataset(data_dir, name):
    d = np.load(Path(data_dir) / CACHE_NAME / (name + ".npz"), allow_pickle=True)
    return d["X"], d["y"]


# --------------------------------------------------------------------------- #
# Clusterlets partition
# --------------------------------------------------------------------------- #
def build_fragments(X, y, fragments_per_class=4, seed=42):
    """K-Means inside each class -> list of (class, indices)."""
    rng = np.random.RandomState(seed)
    frags = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        f = int(min(fragments_per_class, len(idx)))
        if f <= 1:
            frags.append((int(c), idx))
            continue
        km = KMeans(n_clusters=f, n_init=10, random_state=int(rng.randint(1 << 30))).fit(X[idx])
        for g in range(f):
            sub = idx[km.labels_ == g]
            if len(sub):
                frags.append((int(c), sub))
    return frags


def assign_fragments(frags, n_clients, seed=42):
    """Greedy balanced assignment; zero sample overlap by construction."""
    rng = np.random.RandomState(seed)
    order = sorted(range(len(frags)), key=lambda i: -len(frags[i][1]))
    loads = np.zeros(n_clients, dtype=np.int64)
    buckets = [[] for _ in range(n_clients)]
    held = [Counter() for _ in range(n_clients)]
    for i in order:
        c, sub = frags[i]
        cand = sorted(range(n_clients), key=lambda m: (loads[m], held[m][c]))
        if len(cand) > 1 and rng.rand() < 0.15:
            cand[0], cand[1] = cand[1], cand[0]
        m = cand[0]
        buckets[m].append(sub)
        loads[m] += len(sub)
        held[m][c] += 1
    parts = [np.sort(np.concatenate(b)) if b else np.array([], dtype=int) for b in buckets]
    stats = {"sizes": [int(len(p)) for p in parts],
             "classes_per_client": [int(len(held[m])) for m in range(n_clients)],
             "clients_per_class": {int(c): int(sum(1 for m in range(n_clients) if held[m][c]))
                                   for c in sorted({int(f[0]) for f in frags})},
             "n_fragments": len(frags)}
    return parts, stats


def make_clusterlet_partition(X, y, n_clients=10, fragments_per_class=4, seed=42):
    parts, stats = assign_fragments(build_fragments(X, y, fragments_per_class, seed), n_clients, seed)
    all_idx = np.concatenate(parts)
    stats["overlap_samples"] = int(len(all_idx) - len(np.unique(all_idx)))
    return parts, stats


# --------------------------------------------------------------------------- #
# client construction (what the author's main.py expects)
# --------------------------------------------------------------------------- #
def _loader(X, y, batch_size):
    return DataLoader(TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                                    torch.as_tensor(y, dtype=torch.long)),
                      batch_size=batch_size, shuffle=True)


def _author_csv_once(X, y, path: Path):
    if path.exists():
        return path
    import pandas as pd
    cols = ["f%d" % i for i in range(X.shape[1])] + ["class"]
    df = pd.DataFrame(np.column_stack([X.astype(np.float64), y.astype(np.int64)]), columns=cols)
    df["class"] = y.astype(np.int64)
    df.to_csv(path, index=False)
    return path


def make_clients(dataset, mode, n_clients, batch_size, data_dir, source_copy, seed=42):
    """Return (K, loaders, (X, y), sizes, stats).

    mode = 'clusterlet' : zero sample overlap, features standardised on the subset
    mode = 'author'     : the author's own make_noniid_clients on a CSV of the same subset
    """
    X, y = load_dataset(data_dir, dataset)
    k = int(len(np.unique(y)))

    if mode == "clusterlet":
        Xs, _ = standardize(X)
        parts, stats = make_clusterlet_partition(Xs, y, n_clients, seed=seed)
        stats["overlap_samples"] = int(len(np.concatenate(parts)) -
                                       len(np.unique(np.concatenate(parts))))
        loaders = [_loader(Xs[p], y[p], batch_size) for p in parts]
        return (k, loaders,
                (torch.as_tensor(Xs, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)),
                [int(len(p)) for p in parts], stats)

    if mode == "author":
        csv = _author_csv_once(X, y, Path(data_dir) / CACHE_NAME / (dataset + ".csv"))
        from dataloader import make_noniid_clients        # author module on sys.path
        k2, loaders, (Xt, yt), sizes = make_noniid_clients(
            str(csv), n_clients, batch_size, dataset, seed=seed)
        return k2, loaders, (Xt, yt), [int(s) for s in sizes], {"mode": "author"}

    raise ValueError("unknown mode: %r" % mode)


def half_split(y, seed=0):
    """Class-stratified half/half split, used by the matched coverage metric."""
    rng = np.random.RandomState(seed)
    ref, qry = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        h = len(idx) // 2
        ref.append(idx[:h])
        qry.append(idx[h:])
    return np.sort(np.concatenate(ref)), np.sort(np.concatenate(qry))

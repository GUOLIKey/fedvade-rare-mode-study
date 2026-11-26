"""Fashion-MNIST preparation: download, flatten, long-tail construction, caching.

Everything is written under this experiment directory.  Nothing in FedVaDE_source/
is touched.

Outputs (cache/*.npz):
    X  float32 (N, 784)   already standardized
    y  int64   (N,)
    plus a metadata dict in 'meta'
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
RAW = HERE / "raw"


def _read_idx(path: Path) -> np.ndarray:
    """Minimal IDX parser (no torch dependency)."""
    import gzip
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        ndim = magic & 0xFF
        dims = [int.from_bytes(f.read(4), "big") for _ in range(ndim)]
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return buf.reshape(dims)


def load_fmnist_raw() -> tuple[np.ndarray, np.ndarray]:
    """Return the flattened, [0,1]-scaled train split from the raw IDX files.

    torchvision is only used to fetch the files once; parsing is pure numpy,
    because the local torch/numpy/python combination segfaults on the loader.
    """
    base = RAW / "FashionMNIST" / "raw"
    img = base / "train-images-idx3-ubyte"
    lab = base / "train-labels-idx1-ubyte"
    if not img.exists() or not lab.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        from torchvision import datasets  # download-only fallback
        datasets.FashionMNIST(root=str(RAW), train=True, download=True)
    X = _read_idx(img).reshape(-1, 784).astype(np.float32) / 255.0
    y = _read_idx(lab).astype(np.int64)
    return X, y


def stratified_subsample(X, y, n_target, seed=0):
    """Keep the class proportions of the input while taking roughly n_target rows."""
    rng = np.random.RandomState(seed)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        share = len(idx) / len(y)
        m = min(len(idx), max(1, int(round(n_target * share))))
        keep.append(rng.choice(idx, m, replace=False))
    sel = np.sort(np.concatenate(keep))
    return X[sel], y[sel]


def make_long_tailed(X, y, rho, n_target=10000, seed=0):
    """Exponential long tail: class i keeps round(n_max * rho^(-i/(k-1))) samples.

    Class order is by descending original frequency so index 0 is the head.
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    counts = np.array([int((y == c).sum()) for c in classes])
    order = classes[np.argsort(-counts)]
    k = len(order)
    head = int(counts.max())
    keep = {}
    for rank, c in enumerate(order):
        target = max(1, int(round(head * (1.0 / rho) ** (rank / max(1, k - 1)))))
        idx = np.where(y == c)[0]
        if target < len(idx):
            idx = rng.choice(idx, target, replace=False)
        keep[int(c)] = idx
    sel = np.sort(np.concatenate(list(keep.values())))
    Xs, ys = X[sel], y[sel]
    if len(ys) > n_target:
        Xs, ys = stratified_subsample(Xs, ys, n_target, seed=seed)
    return Xs, ys


def standardize(X_train, X_ref=None):
    """Fit on X_train (the federated data), apply to both.  Idempotent per name."""
    sc = StandardScaler().fit(X_train)
    out = sc.transform(X_train).astype(np.float32)
    if X_ref is None:
        return out, None
    return out, sc.transform(X_ref).astype(np.float32)


def build(name: str, rho: float | None, n_target: int, seed: int = 0) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{name}.npz"
    if out.exists():
        return out
    X, y = load_fmnist_raw()
    if rho is None:
        Xs, ys = stratified_subsample(X, y, n_target, seed=seed)
    else:
        Xs, ys = make_long_tailed(X, y, rho, n_target=n_target, seed=seed)
    counts = {int(c): int(n) for c, n in zip(*np.unique(ys, return_counts=True))}
    meta = {"name": name, "rho": rho, "n": int(len(ys)), "n_target": n_target,
            "seed": seed, "class_counts": counts,
            "source": "FashionMNIST train split (IDX), flattened 28x28, scaled to [0,1]. "
                      "Stored RAW; consumers standardize."}
    np.savez_compressed(out, X=Xs, y=ys, meta=json.dumps(meta))
    (CACHE / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    return out


def load(name: str):
    """Raw [0,1]-scaled features (unstandardized) plus labels and metadata."""
    d = np.load(CACHE / f"{name}.npz", allow_pickle=True)
    return d["X"], d["y"], json.loads(str(d["meta"]))


def load_std(name: str):
    """Standardized features, scaler fit on the whole subset (author convention)."""
    X, y, meta = load(name)
    Xs, _ = standardize(X)
    return Xs, y, meta


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    a = ap.parse_args()
    for name, rho in (("fmnist_bal", None), ("fmnist_lt10", 10.0), ("fmnist_lt50", 50.0)):
        p = build(name, rho, a.n)
        X, y, meta = load(name)
        print("%-12s -> %s  N=%d  counts=%s" % (name, p.name, len(y), meta["class_counts"]))

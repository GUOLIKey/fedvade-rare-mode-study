"""Client construction for Fashion-MNIST under two partitions.

Returns exactly the tuple the author's main.py expects:

    (n_clusters, [DataLoader, ...], (X_tensor, y_tensor), sizes)

Two modes
---------
clusterlet : our Clusterlets partition (K-Means fragments inside each class,
             zero sample overlap).  Features are standardized on the whole
             subset first, matching the author's `dataloader.py` convention.
author     : the author's own `make_noniid_clients`, called verbatim on a CSV
             written from the same raw subset.  This is the control that
             isolates the effect of the partition itself.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import fmnist_prep
from clusterlet_partition import make_clusterlet_partition


def _loader(X, y, batch_size):
    return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32),
                                    torch.tensor(y, dtype=torch.long)),
                      batch_size=batch_size, shuffle=True)


def _write_csv_once(X, y, path: Path):
    """Raw [0,1] features -> CSV, written once and reused by every run."""
    if path.exists():
        return path
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["f%d" % i for i in range(X.shape[1])] + ["class"]
    frame = pd.DataFrame(np.column_stack([X.astype(np.float64), y.astype(np.int64)]),
                         columns=cols)
    frame["class"] = y.astype(np.int64)
    frame.to_csv(path, index=False)
    return path


def make_clients(name, mode, n_clients, batch_size, data_dir, source_copy, seed=42):
    X, y, meta = fmnist_prep.load(name)          # raw, [0,1]
    k = int(len(np.unique(y)))

    if mode == "clusterlet":
        Xs, _ = fmnist_prep.standardize(X)
        parts, stats = make_clusterlet_partition(Xs, y, n_clients=n_clients, seed=seed)
        loaders = [_loader(Xs[p], y[p], batch_size) for p in parts]
        sizes = [int(len(p)) for p in parts]
        overload = int(len(np.concatenate(parts)) - len(np.unique(np.concatenate(parts))))
        stats["overlap_samples"] = overload
        return (k,
                loaders,
                (torch.tensor(Xs, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
                sizes,
                stats)

    if mode == "author":
        csv = _write_csv_once(X, y, Path(data_dir) / "cache" / (name + ".csv"))
        from dataloader import make_noniid_clients       # imported from the run's source copy
        k2, loaders, (Xt, yt), sizes = make_noniid_clients(
            str(csv), n_clients, batch_size, name, seed=seed)
        stats = {"mode": "author", "overlap": "clients overlap by construction"}
        return k2, loaders, (Xt, yt), [int(s) for s in sizes], stats

    raise ValueError("unknown mode: %r" % mode)

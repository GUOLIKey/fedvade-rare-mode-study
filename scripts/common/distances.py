"""Nearest-neighbour helpers.

All of them use the Gram-matrix identity instead of broadcasting
(|a|^2 + |b|^2 - 2 a.b).  The broadcasting version materialises a
chunk x N x D array, which for N=10000, D=784 is ~8 GB and hangs the machine.
"""
from __future__ import annotations

import numpy as np


def min_dist_to_set(A, B, chunk: int = 1024):
    """Distance from each row of A to the nearest row of B."""
    A = np.ascontiguousarray(A, np.float32)
    B = np.ascontiguousarray(B, np.float32)
    sqB = (B * B).sum(1)
    out = np.empty(len(A))
    for i in range(0, len(A), chunk):
        j = min(len(A), i + chunk)
        a = A[i:j]
        d = (a * a).sum(1)[:, None] + sqB[None, :] - 2.0 * (a @ B.T)
        out[i:j] = np.sqrt(np.maximum(d.min(1), 0.0))
    return out


def min_dist_self(A, chunk: int = 1024):
    """Nearest neighbour inside A, the point itself excluded."""
    A = np.ascontiguousarray(A, np.float32)
    sq = (A * A).sum(1)
    out = np.empty(len(A))
    for i in range(0, len(A), chunk):
        j = min(len(A), i + chunk)
        d = sq[i:j, None] + sq[None, :] - 2.0 * (A[i:j] @ A.T)
        d[np.arange(j - i), i + np.arange(j - i)] = np.inf
        out[i:j] = np.sqrt(np.maximum(d.min(1), 0.0))
    return out


def nearest_index(A, B, chunk: int = 1024):
    """Index in B of the nearest row to each row of A."""
    A = np.ascontiguousarray(A, np.float32)
    B = np.ascontiguousarray(B, np.float32)
    sqB = (B * B).sum(1)
    out = np.empty(len(A), dtype=np.int64)
    for i in range(0, len(A), chunk):
        j = min(len(A), i + chunk)
        a = A[i:j]
        d = (a * a).sum(1)[:, None] + sqB[None, :] - 2.0 * (a @ B.T)
        out[i:j] = d.argmin(1)
    return out


def knn_purity(X, y, chunk: int = 2000):
    """1-NN same-class fraction, overall and per class, plus the neighbour index."""
    X = np.ascontiguousarray(X, np.float32)
    sq = (X * X).sum(1)
    n = len(X)
    nn = np.empty(n, dtype=np.int32)
    for i in range(0, n, chunk):
        j = min(n, i + chunk)
        d = sq[i:j, None] + sq[None, :] - 2.0 * (X[i:j] @ X.T)
        d[np.arange(j - i), i + np.arange(j - i)] = np.inf
        nn[i:j] = d.argmin(1)
    per = {}
    for c in np.unique(y):
        sel = y == c
        per[int(c)] = {"n": int(sel.sum()), "share": float(sel.mean()),
                       "purity": float((y[nn[sel]] == c).mean())}
    return float((y[nn] == y).mean()), per, nn

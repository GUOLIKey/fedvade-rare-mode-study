"""Clusterlets partition.

Replicates the *observed properties* of the partitions shipped with FedHCA
(see REPORT.md, section "released partitions"), not their unknown generator:

  * fixed number of clients
  * zero sample overlap -- every sample belongs to exactly one client
  * each class is fragmented into several feature-space sub-clusters and those
    fragments are spread over several clients (the "clusterlet" scenario)

Fragment construction follows the reviewer's correction: K-Means inside each
class, not a random split.  A random split would give every fragment the same
distribution and would not model multi-granularity at all.

This is OUR implementation.  It must be described as "constructed to match the
Clusterlets properties described by FedHCA", never as "FedHCA's partition".
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import KMeans


def build_fragments(X, y, fragments_per_class=4, seed=42):
    """K-Means inside each class -> list of (class, indices, size)."""
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


def assign_fragments(frags, n_clients, min_classes=2, seed=42):
    """Greedy balanced assignment with zero sample overlap.

    Largest fragment first, each going to the client with the fewest samples.
    A tie-break prefers a client that does not yet hold this class, so classes
    spread out instead of piling onto one client.
    """
    rng = np.random.RandomState(seed)
    order = sorted(range(len(frags)), key=lambda i: -len(frags[i][1]))
    loads = np.zeros(n_clients, dtype=np.int64)
    client_idx = [[] for _ in range(n_clients)]
    client_classes = [Counter() for _ in range(n_clients)]

    for i in order:
        c, sub = frags[i]
        cand = sorted(range(n_clients), key=lambda m: (loads[m], client_classes[m][c]))
        if len(cand) > 1 and rng.rand() < 0.15:      # small jitter, keeps determinism per seed
            cand[0], cand[1] = cand[1], cand[0]
        m = cand[0]
        client_idx[m].append(sub)
        loads[m] += len(sub)
        client_classes[m][c] += 1

    out = []
    for m in range(n_clients):
        idx = np.sort(np.concatenate(client_idx[m])) if client_idx[m] else np.array([], dtype=int)
        out.append(idx)

    stats = {
        "sizes": [int(len(a)) for a in out],
        "classes_per_client": [int(len(client_classes[m])) for m in range(n_clients)],
        "clients_per_class": {int(c): int(sum(1 for m in range(n_clients) if client_classes[m][c] > 0))
                              for c in sorted({int(f[0]) for f in frags})},
        "min_classes_constraint_ok": bool(all(len(client_classes[m]) >= min(min_classes, len(np.unique(
            np.concatenate([f[1] for f in frags])))) for m in range(n_clients))),
    }
    return out, stats


def make_clusterlet_partition(X, y, n_clients=10, fragments_per_class=4, seed=42):
    frags = build_fragments(X, y, fragments_per_class=fragments_per_class, seed=seed)
    parts, stats = assign_fragments(frags, n_clients, seed=seed)
    stats["n_fragments"] = len(frags)
    stats["fragments_per_class_requested"] = fragments_per_class
    return parts, stats


def summarize(parts, y):
    """Sanity report: overlap, sizes, per-client class counts."""
    all_idx = np.concatenate(parts)
    overlap = len(all_idx) - len(np.unique(all_idx))
    lines = ["overlap_samples=%d (must be 0)" % overlap,
             "sizes=%s" % [len(p) for p in parts],
             "classes_per_client=%s" % [len(np.unique(y[p])) for p in parts]]
    for m, p in enumerate(parts):
        cls = Counter(y[p].tolist())
        lines.append("  client %2d: n=%4d  classes=%s" % (
            m, len(p), dict(sorted((int(k), int(v)) for k, v in cls.items()))))
    return "\n".join(lines)

"""Clustering metrics shared by every evaluation and diagnostic script."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def cluster_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    d = int(max(y_true.max(), y_pred.max())) + 1
    conf = np.zeros((d, d), dtype=np.int64)
    for a, b in zip(y_true, y_pred):
        conf[int(a), int(b)] += 1
    r, c = linear_sum_assignment(-conf)
    return {"ari": float(adjusted_rand_score(y_true, y_pred)),
            "purity": float(np.max(conf, axis=0).sum() / len(y_true)),
            "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
            "acc": float(conf[r, c].sum() / len(y_true))}

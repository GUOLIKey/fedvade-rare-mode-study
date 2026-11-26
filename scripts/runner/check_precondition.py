"""Precondition check -- run BEFORE any federated training.

Motivated by a past failure: four server-side experiments were run on yeast,
where the rare classes are not separable in the input space at all, so none of
them could ever have shown an effect.  This script answers, cheaply:

  1. per-class 1-nearest-neighbour purity in the raw feature space
       -> is the rare class a real cluster, or noise?
  2. centralized GaussianMixture(K) ARI on the raw features
       -> can a mixture model recover the classes at all?
  3. KMeans(K) ARI as a second cheap reference
  4. clusterlet partition statistics (zero overlap, classes per client)

Nothing here touches FedVaDE_source/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "data"))

import fmnist_prep  # noqa: E402
from clusterlet_partition import make_clusterlet_partition, summarize  # noqa: E402


def knn_purity(X, y, chunk=2000):
    """1-NN same-class fraction, self excluded, via chunked Gram matrix."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    sq = (X * X).sum(1)
    n = len(X)
    nn = np.empty(n, dtype=np.int32)
    for i in range(0, n, chunk):
        j = min(n, i + chunk)
        d = sq[i:j, None] + sq[None, :] - 2.0 * (X[i:j] @ X.T)
        rows = np.arange(j - i)
        d[rows, i + rows] = np.inf           # exclude self
        nn[i:j] = d.argmin(1)
    overall = float((y[nn] == y).mean())
    per_class = {}
    for c in np.unique(y):
        sel = y == c
        per_class[int(c)] = {
            "n": int(sel.sum()),
            "share": float(sel.mean()),
            "purity": float((y[nn[sel]] == c).mean()),
        }
    return overall, per_class


def check(name, rho, n_target, fragments=4, n_clients=10):
    path = fmnist_prep.build(name, rho, n_target)
    X, y, meta = fmnist_prep.load(name)
    k = len(np.unique(y))
    print("=" * 78)
    print("%s   N=%d  k=%d  dim=%d  rho=%s" % (name, len(y), k, X.shape[1], rho))
    print("class counts:", meta["class_counts"])

    t0 = time.time()
    overall, per_class = knn_purity(X, y)
    print("\n[1] 原始特征空间 1-近邻纯度  整体 = %.4f  (%.1fs)" % (overall, time.time() - t0))
    print("    %-6s %8s %8s %10s %s" % ("class", "n", "share", "purity", "verdict"))
    for c, v in per_class.items():
        vs = v["purity"] / v["share"] if v["share"] > 0 else 0.0
        flag = "可分" if vs > 5 else ("弱" if vs > 2 else "不可分")
        print("    %-6d %8d %7.1f%% %10.3f  %s (purity/share=%.1f)" % (
            c, v["n"], 100 * v["share"], v["purity"], flag, vs))

    t0 = time.time()
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    ari_km = adjusted_rand_score(y, km.labels_)
    print("\n[2] 集中式 KMeans(K=%d)  ARI = %.4f  (%.1fs)" % (k, ari_km, time.time() - t0))

    t0 = time.time()
    gm = GaussianMixture(n_components=k, covariance_type="diag", random_state=42,
                         reg_covar=1e-3, n_init=1, max_iter=100).fit(X)
    ari_gm = adjusted_rand_score(y, gm.predict(X))
    print("[3] 集中式 GMM(K=%d, diag)  ARI = %.4f  (%.1fs)" % (k, ari_gm, time.time() - t0))

    parts, stats = make_clusterlet_partition(X, y, n_clients=n_clients,
                                             fragments_per_class=fragments)
    print("\n[4] clusterlet 划分 (M=%d, 每类 %d 碎片, 类内 KMeans)" % (n_clients, fragments))
    print(summarize(parts, y))
    print("    每客户端类数:", stats["classes_per_client"])
    print("    每类被几个客户端持有:", stats["clients_per_class"])

    return {"name": name, "n": int(len(y)), "k": int(k), "rho": rho,
            "knn_overall": overall, "per_class": per_class,
            "kmeans_ari": float(ari_km), "gmm_ari": float(ari_gm), "partition": stats}


def main():
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    results = []
    for name, rho in (("fmnist_bal", None), ("fmnist_lt10", 10.0)):
        results.append(check(name, rho, n_target))
    print("\n" + "=" * 78)
    print("汇总")
    print("%-12s %6s %10s %10s %10s" % ("dataset", "k", "1NN", "KMeans", "GMM"))
    for r in results:
        print("%-12s %6d %10.4f %10.4f %10.4f" % (r["name"], r["k"], r["knn_overall"],
                                                  r["kmeans_ari"], r["gmm_ari"]))


if __name__ == "__main__":
    main()

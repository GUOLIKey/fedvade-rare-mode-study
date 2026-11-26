"""Local GMM construction for the client side: DPGMM, or a same-K control.

Both modes go through the same DPGMM fit so the component count is identical
between them; the only difference is whether the parameters come from the DP
fit or from an ordinary GaussianMixture re-fit at that K.  That isolates
"how the local mixture is modelled" from "how many components it has".
"""
from __future__ import annotations

import numpy as np
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

REG_COVAR = 1e-6


def fit_dpgmm(mu, wcp, n_components=15, prune=0.01, random_state=42):
    bgm = BayesianGaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=wcp,
        random_state=random_state,
        reg_covar=REG_COVAR,
        max_iter=500,
    ).fit(mu)
    w = np.asarray(bgm.weights_, dtype=np.float64)
    keep = np.flatnonzero(w > prune)
    if keep.size == 0:                       # never return zero components
        keep = np.array([int(np.argmax(w))])
    w = w[keep]
    w = w / w.sum()
    return w, np.asarray(bgm.means_)[keep], np.asarray(bgm.covariances_)[keep]


def fit_same_k_gmm(mu, k, random_state=42):
    gm = GaussianMixture(n_components=int(k), covariance_type="diag",
                         random_state=random_state, reg_covar=REG_COVAR).fit(mu)
    w = np.asarray(gm.weights_, dtype=np.float64)
    w = w / w.sum()
    return w, np.asarray(gm.means_), np.asarray(gm.covariances_)


def fit_local(mu, mode, wcp, prune=0.01, n_components=15, random_state=42):
    """Returns (weights, means, covariances, retained_K)."""
    w, m, c = fit_dpgmm(mu, wcp, n_components=n_components, prune=prune,
                        random_state=random_state)
    if mode == "dpgmm":
        return w, m, c, len(w)
    if mode == "samek_gmm":
        w2, m2, c2 = fit_same_k_gmm(mu, len(w), random_state=random_state)
        return w2, m2, c2, len(w)
    raise ValueError("unknown mode: %r" % mode)

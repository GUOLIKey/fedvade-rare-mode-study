"""Patch the copied model.py so the local GMM is built by our fitter.

Replaces, inside pre_train():

    K_l = len(np.unique(local_y))
    gmm = GaussianMixture(n_components=K_l, ...)
    gmm.fit(local_mu)
    pi_ = torch.tensor(gmm.weights_, ...)
    c['vade'].rho_.data = torch.log(pi_ + epsilon)
    c['vade'].mu_c.data = torch.tensor(gmm.means_, ...)
    c['vade'].log_sigma2_c.data = torch.log(torch.tensor(gmm.covariances_, ...))

with a call to `_fit_local`, which the runner injects into the exec namespace.
The author's line that reads the labels is kept only for logging; K no longer
comes from the labels in the dpgmm / samek_gmm modes.
"""
from __future__ import annotations

from pathlib import Path

OLD_BLOCK = """            K_l = len(np.unique(local_y))
            gmm = GaussianMixture(n_components=K_l, covariance_type='diag', random_state=42)
            gmm.fit(local_mu)

            pi_ = torch.tensor(gmm.weights_, dtype=torch.float, device=device)
            c['vade'].rho_.data = torch.log(pi_ + epsilon)
            c['vade'].mu_c.data = torch.tensor(gmm.means_, dtype=torch.float, device=device)
            c['vade'].log_sigma2_c.data = torch.log(torch.tensor(gmm.covariances_, dtype=torch.float, device=device))
"""

NEW_BLOCK = """            _w, _m, _s, _k = _fit_local(local_mu, local_y)
            _record_k(c['client_id'], _k)
            pi_ = torch.tensor(_w, dtype=torch.float, device=device)
            c['vade'].rho_.data = torch.log(pi_ + epsilon)
            c['vade'].mu_c.data = torch.tensor(_m, dtype=torch.float, device=device)
            c['vade'].log_sigma2_c.data = torch.log(torch.tensor(_s, dtype=torch.float, device=device))
"""


def apply(path: Path) -> None:
    src = path.read_text()
    n = src.count(OLD_BLOCK)
    if n != 1:
        raise ValueError("expected exactly one local-GMM block in model.py, found %d" % n)
    path.write_text(src.replace(OLD_BLOCK, NEW_BLOCK))

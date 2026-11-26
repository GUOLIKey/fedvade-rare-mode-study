"""Build the author's VaDE from a saved state and query its Mixture-of-Gaussians."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch


def build_vade(state: dict, hid_dim: int, alpha: float, input_dim: int):
    """Instantiate the author's VaDE with the shapes found in `state`."""
    from model import VaDE  # author module, on sys.path
    m = VaDE(SimpleNamespace(input_dim=int(input_dim), hid_dim=int(hid_dim),
                             nClusters=len(state["rho_"]), alpha=float(alpha),
                             cuda=False))
    m.load_state_dict(state, strict=True)
    m.eval()
    return m


def load_server_state(run_dir, name="last_checkpoint.pt"):
    """Read the server state dict and its metadata from a run directory."""
    import json
    from pathlib import Path
    run_dir = Path(run_dir)
    path = run_dir / name
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = json.loads((run_dir / "config.json").read_text())
    return payload["server"], payload, cfg


def hid_dim_of(cfg: dict) -> int:
    return int(cfg.get("hid_dim", cfg.get("latent_dim", 10)))


def encode_mean(model, X) -> np.ndarray:
    """Posterior mean only -- no sampling, so the result is deterministic."""
    with torch.no_grad():
        mu, _ = model.encoder(torch.as_tensor(X, dtype=torch.float32))
    return mu.numpy()


def responsibilities(model, X):
    """Posterior mean and hard component assignment under the model's own GMM."""
    with torch.no_grad():
        mu, _ = model.encoder(torch.as_tensor(X, dtype=torch.float32))
        logp = model.gaussian_pdfs_log(mu, model.mu_c, model.log_sigma2_c) \
            + torch.log(torch.softmax(model.rho_, 0)).unsqueeze(0)
    return mu.numpy(), logp.argmax(1).numpy()


def component_table(state, assign, y, n_classes):
    """Per-component: weight, size, dominant classes."""
    pi = torch.softmax(state["rho_"], 0).detach().numpy()
    rows = []
    for c in range(len(pi)):
        sel = assign == c
        n = int(sel.sum())
        row = {"component": c, "pi": round(float(pi[c]), 4), "n": n,
               "share_of_data": round(n / len(y), 4),
               "mean_log_sigma2": round(float(state["log_sigma2_c"][c].mean()), 3)}
        if n:
            bc = np.bincount(y[sel], minlength=n_classes)
            row["top_classes"] = [[int(t), int(bc[t])] for t in bc.argsort()[::-1][:3] if bc[t] > 0]
        rows.append(row)
    return rows

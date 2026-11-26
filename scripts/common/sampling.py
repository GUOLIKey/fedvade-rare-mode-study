"""Draw samples from a server-side Gaussian mixture and decode them."""
from __future__ import annotations

import numpy as np
import torch


def sample_latents(model, n: int, seed: int = 0) -> np.ndarray:
    """z ~ sum_k pi_k N(mu_k, diag(sigma2_k)) under the model's own GMM."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    pi = torch.softmax(model.rho_, 0).detach().numpy()
    mu = model.mu_c.detach().numpy()
    var = torch.exp(model.log_sigma2_c).detach().numpy()
    comp = np.random.choice(len(pi), size=n, p=pi)
    return mu[comp] + np.sqrt(var[comp]) * np.random.randn(n, mu.shape[1])


def decode(model, z) -> np.ndarray:
    with torch.no_grad():
        return model.decoder(torch.as_tensor(z, dtype=torch.float32)).numpy()


def generate(model, n: int, seed: int = 0) -> np.ndarray:
    """Sample and decode in one call."""
    return decode(model, sample_latents(model, n, seed))

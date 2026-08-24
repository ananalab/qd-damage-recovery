"""ITE (M-BOA) and baselines on a synthetic problem with a known answer."""

import numpy as np
import pytest

from qd_damage.baselines import run_best_intact, run_random, run_top_k
from qd_damage.ite import GaussianProcess, Normalizer, matern52, run_ite

PARAMS = dict(kappa=0.05, alpha=0.9, lengthscale=0.4, noise_variance=0.001)


def synthetic_problem():
    """30x30 grid in [0,1]^2. Before damage, two families of good gaits: around (0.8, 0.8) (performance 1.0)
    and around (0.2, 0.3) (performance 0.8). The damage destroys the first family and spares the second, so
    the new optimum is at (0.2, 0.3). As on the robot, damage only degrades (true <= prior)."""
    g = np.linspace(0, 1, 30)
    x = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    d_a, d_b = ((x - [0.8, 0.8]) ** 2).sum(1), ((x - [0.2, 0.3]) ** 2).sum(1)
    prior = np.maximum(1.0 - 3.0 * d_a, 0.8 - 3.0 * d_b)
    damage = 0.9 * np.exp(-d_a / (2 * 0.15**2))
    true = prior - damage
    return x, prior, true


def test_matern_kernel_properties():
    x = np.random.default_rng(0).random((5, 3))
    k = matern52(x, x, 0.4)
    np.testing.assert_allclose(np.diag(k), 1.0)
    np.testing.assert_allclose(k, k.T)
    assert np.all(np.linalg.eigvalsh(k) > -1e-9)


def test_gp_interpolates_and_is_uncertain_far_away():
    x = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.2]])
    y = np.array([0.3, -0.2, 0.1])
    gp = GaussianProcess(lengthscale=0.4, noise_variance=1e-6).fit(x, y)
    mean, std = gp.predict(x)
    np.testing.assert_allclose(mean, y, atol=1e-3)
    assert np.all(std < 0.01)
    _, std_far = gp.predict(np.array([[5.0, 5.0]]))
    assert std_far[0] > 0.99



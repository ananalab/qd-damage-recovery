"""ITE (M-BOA) and baselines on a synthetic problem with a known answer."""

import numpy as np
import pytest

from qd_damage.baselines import run_best_intact, run_random, run_top_k
from qd_damage.ite import GaussianProcess, Normalizer, matern52, run_ite

PARAMS = dict(kappa=0.05, alpha=0.9, lengthscale=0.4, noise_variance=0.001)


def test_matern_kernel_properties():
    x = np.random.default_rng(0).random((5, 3))
    k = matern52(x, x, 0.4)
    np.testing.assert_allclose(np.diag(k), 1.0)
    np.testing.assert_allclose(k, k.T)
    assert np.all(np.linalg.eigvalsh(k) > -1e-9)



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


def test_ite_finds_new_optimum_quickly():
    x, prior, true = synthetic_problem()
    rng = np.random.default_rng(0)
    history = run_ite(x, prior, lambda i: true[i] + 0.01 * rng.normal(), Normalizer(0.0, 1.0), max_trials=20, **PARAMS)
    assert history.stop_trial is not None and history.stop_trial <= 15
    found = true[history.best_cell(history.stop_trial)]
    assert found >= 0.9 * true.max()


def test_ite_beats_top_k_and_random_after_10_trials():
    x, prior, true = synthetic_problem()
    rng = np.random.default_rng(1)
    trial = lambda i: true[i] + 0.01 * rng.normal()  # noqa: E731
    ite = run_ite(x, prior, trial, Normalizer(0.0, 1.0), max_trials=10, **PARAMS)
    top = run_top_k(prior, trial, 10)
    rand = np.mean([true[run_random(len(x), trial, 10, np.random.default_rng(s)).best_cell(10)] for s in range(20)])
    assert true[ite.best_cell(10)] > true[top.best_cell(10)]
    assert true[ite.best_cell(10)] > rand


def test_baselines_interface():
    prior = np.array([0.1, 0.9, 0.5])
    hist = run_best_intact(prior, lambda i: float(i))
    assert hist.cells == [1]
    assert run_top_k(prior, lambda i: float(i), 3).cells == [1, 2, 0]
    assert sorted(run_random(3, lambda i: 0.0, 3, np.random.default_rng(0)).cells) == [0, 1, 2]


@pytest.mark.parametrize("f", [0.0, 810.0, 1500.0])
def test_normalizer_roundtrip(f):
    n = Normalizer(f0=1058.7, scale=300.0)
    assert n.inverse(n(f)) == pytest.approx(f)
    assert n(1358.7) == pytest.approx(1.0)

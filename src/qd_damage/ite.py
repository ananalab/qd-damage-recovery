"""Intelligent Trial & Error: M-BOA (Map-Based Bayesian Optimisation Algorithm), Cully et al., Nature 2015.

The repertoire gives, for each cell, the performance measured on the intact robot (the prior). After damage,
we try a gait, observe its performance, and a Gaussian process corrects the predictions of all similar gaits
(close in descriptor space). The next trial maximises an upper confidence bound, and the search stops as soon
as a tested gait reaches a fraction alpha of the best performance still expected.

Plain numpy: there are at most a few dozen trials. Performances are normalised (see `Normalizer`) so that the
parameters of Cully et al. (kappa, noise, length-scale) keep their meaning.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


def matern52(a: np.ndarray, b: np.ndarray, lengthscale: float) -> np.ndarray:
    """Matern 5/2 kernel (unit variance) between point sets of shape (n, d) and (m, d)."""
    d = np.sqrt(np.maximum(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1), 0.0)) / lengthscale
    return (1.0 + np.sqrt(5.0) * d + 5.0 / 3.0 * d**2) * np.exp(-np.sqrt(5.0) * d)


class GaussianProcess:
    """Gaussian process on the residuals (observed - prior), kernel signal_variance x Matern 5/2."""

    def __init__(self, lengthscale: float, noise_variance: float, signal_variance: float = 1.0):
        self.lengthscale = lengthscale
        self.noise_variance = noise_variance
        self.signal_variance = signal_variance
        self.x = np.zeros((0, 0))
        self.residuals = np.zeros(0)

    def kernel(self, a, b):
        return self.signal_variance * matern52(a, b, self.lengthscale)

    def fit(self, x: np.ndarray, residuals: np.ndarray):
        self.x, self.residuals = np.asarray(x, float), np.asarray(residuals, float)
        k = self.kernel(self.x, self.x) + self.noise_variance * np.eye(len(self.x))
        self._chol = np.linalg.cholesky(k)
        self._alpha = np.linalg.solve(self._chol.T, np.linalg.solve(self._chol, self.residuals))
        return self

    def predict(self, x: np.ndarray):
        """Mean and standard deviation of the residual at each point of x."""
        x = np.asarray(x, float)
        if len(self.x) == 0:
            return np.zeros(len(x)), np.full(len(x), np.sqrt(self.signal_variance))
        k_star = self.kernel(x, self.x)  # (m, n)
        mean = k_star @ self._alpha
        v = np.linalg.solve(self._chol, k_star.T)
        var = np.maximum(self.signal_variance - (v**2).sum(0), 1e-12)
        return mean, np.sqrt(var)

    def log_marginal_likelihood(self, x: np.ndarray, residuals: np.ndarray) -> float:
        """log p(residuals | hyper-parameters), used to choose the hyper-parameters on development data."""
        self.fit(x, residuals)
        n = len(residuals)
        return float(
            -0.5 * self.residuals @ self._alpha - np.log(np.diag(self._chol)).sum() - 0.5 * n * np.log(2 * np.pi)
        )


@dataclass
class Normalizer:
    """Normalised performance p = (F - f0) / scale.

    f0 is the return of a standing robot and scale a fixed scale shared by all repertoires
    (configs/adaptation.yaml): p is about 0 for a robot that does not move and about 1 for the best known gait,
    like the speeds in m/s of Cully et al., and the noise of one episode has the same variance as theirs (~0.001).
    """

    f0: float
    scale: float

    def __call__(self, f):
        return (np.asarray(f, float) - self.f0) / self.scale

    def inverse(self, p):
        return np.asarray(p, float) * self.scale + self.f0



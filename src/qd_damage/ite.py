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



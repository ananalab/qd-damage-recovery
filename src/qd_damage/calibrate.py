"""Choice of the ITE Gaussian-process hyper-parameters by maximum marginal likelihood.

Usage:
    python -m qd_damage.calibrate --oracles results/oracle/me_seed0_demo results/oracle/pgame_seed0_demo ...
    python -m qd_damage.calibrate --joint-usage --oracles ...     # same, on the joint-usage descriptor

Data: development repertoires only (the 200-iteration demo runs), never those of the final campaign. For each
(repertoire, damage) pair, the oracle gives normalised residuals r = (damaged - intact) / scale on a random subset
of cells; the log marginal likelihood is summed over all pairs and the best (length-scale, signal variance,
noise variance) on a grid is kept. The chosen values are then frozen in configs/adaptation.yaml.

The script also reports the likelihood of a model without any correlation between cells (independent noise). The
gap between the two measures how much one trial tells about the other cells, i.e. how much ITE can generalise.
"""

import numpy as np

GRID = {
    "lengthscale": [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2, 1.6],
    "signal_variance": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0],
    "noise_variance": [0.001, 0.003, 0.01, 0.02, 0.04],
}
CULLY = (0.4, 1.0, 0.001)  # (length-scale, signal variance, noise variance) of Cully et al. 2015


def independent_log_likelihood(sets) -> float:
    """Log-likelihood of independent Gaussian residuals (no correlation between cells), variance fitted per pair."""
    total = 0.0
    for _, _, _, r in sets:
        var = float(np.mean(r**2))
        total += -0.5 * len(r) * (np.log(2 * np.pi * var) + 1.0)
    return total


def calibrate(oracle_dirs, n_cells: int = 300, seed: int = 0, joint_usage: bool = False):
    sets = residual_sets(oracle_dirs, n_cells, seed, joint_usage)
    results = []
    for ls, sv, nv in itertools.product(GRID["lengthscale"], GRID["signal_variance"], GRID["noise_variance"]):
        gp = GaussianProcess(ls, nv, sv)
        ll = sum(gp.log_marginal_likelihood(x, r) for _, _, x, r in sets)
        results.append((ll, ls, sv, nv))
    results.sort(reverse=True)
    return results, sets



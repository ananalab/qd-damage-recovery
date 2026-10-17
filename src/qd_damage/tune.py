"""Choice of the adaptation hyper-parameters on the DEVELOPMENT repertoires only (200-iteration runs, never used for
the results).

Usage:
    python -m qd_damage.tune --gp --banks results/bank/*_demo          (Gaussian-process fit, per robot)
    python -m qd_damage.tune --search --banks results/bank/*_demo      (kappa of ITE, settings of ITE-SD)

--gp fits (length-scale, variance, noise) of the Matern kernel on each descriptor by maximum marginal likelihood
of the residuals (damaged - intact) under the simulated training damages, as in the first campaign but without
looking at any test damage. --search replays the in-family test damages of the development banks for a small grid
of settings and reports the mean u after 3, 5 and 10 trials. The chosen values are copied by hand into
configs/adaptation.yaml (section offline) and frozen.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")  # many small matrices: one thread per process (see qd_damage.offline)

import argparse  # noqa: E402
import itertools  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from qd_damage.ite import GaussianProcess  # noqa: E402
from qd_damage.offline import adapt_bank, load_bank, offline_config  # noqa: E402

LENGTHSCALES = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
VARIANCES = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
NOISES = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def fit_gp(banks: list, descriptor: str, n_cells: int = 300, seed: int = 0) -> dict:
    """Grid search of the Matern hyper-parameters maximising the summed log marginal likelihood."""
    rng = np.random.default_rng(seed)
    data = []
    for bank in banks:
        x_all = bank["descriptors"] if descriptor == "feet" else bank["joint_usage"]
        for response in bank["train"][:16]:
            cells = rng.choice(len(x_all), min(n_cells, len(x_all)), replace=False)
            data.append((x_all[cells], (response - bank["prior"])[cells]))
    best = None
    for ls, var, noise in itertools.product(LENGTHSCALES, VARIANCES, NOISES):
        gp = GaussianProcess(ls, noise, var)
        ll = sum(gp.log_marginal_likelihood(x, r) for x, r in data) / len(data)
        if best is None or ll > best[0]:
            best = (ll, {"lengthscale": ls, "variance": var, "noise": noise})
    # Independent cells (no correlation, one global variance like the GP) as a reference: what the kernel adds.
    def iid(v):
        return np.mean([-0.5 * (r @ r / v + len(r) * np.log(2 * np.pi * v)) for _, r in data])

    indep = max(iid(v) for v in np.geomspace(1e-3, 10, 41))
    return {**best[1], "loglik": round(best[0], 1), "loglik_independent": round(float(indep), 1)}


def search(bank_dirs: list, grid: dict, cfg: dict, n_repeats: int = 5) -> list:
    """Mean u after 3, 5, 10 trials on the in-family damages of the development banks, for each setting."""
    out = []
    for name, overrides in grid.items():
        c = {**cfg, **{k: ({**cfg[k], **v} if isinstance(v, dict) else v) for k, v in overrides.items()}}
        method = "itesd" if "itesd" in overrides else "ite_norep"
        scores = defaultdict(list)
        for b in bank_dirs:
            for row in adapt_bank(b, [method], n_repeats, c, groups=("id",)):
                if row[8] in (3, 5, 10):
                    scores[row[8]].append(row[9])
        out.append((name, {k: round(float(np.mean(v)), 4) for k, v in sorted(scores.items())}))
        print(name, out[-1][1], flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--banks", nargs="+", required=True)
    parser.add_argument("--gp", action="store_true")
    parser.add_argument("--search", action="store_true")
    args = parser.parse_args()
    by_robot = defaultdict(list)
    for b in args.banks:
        bank = load_bank(Path(b))
        by_robot[bank["info"]["robot"]].append((b, bank))
    cfg = offline_config()
    for robot, items in by_robot.items():
        if args.gp:
            for descriptor in ("feet", "joint"):
                print(robot, descriptor, fit_gp([bank for _, bank in items], descriptor), flush=True)
        if args.search:
            grid = {f"ite kappa={k}": {"kappa": {"ite": k}} for k in (0.05, 0.3, 1.0, 2.0)}
            for kappa, nugget, noise in itertools.product((0.3, 1.0, 2.0), (0.001, 0.005, 0.02), (0.001, 0.005, 0.02)):
                grid[f"itesd kappa={kappa} nugget={nugget} noise={noise}"] = {
                    "itesd": {"kappa": kappa, "nugget": nugget, "noise": noise}
                }
            print(robot, flush=True)
            search([b for b, _ in items], grid, cfg)


if __name__ == "__main__":
    main()

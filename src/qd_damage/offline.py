"""Adaptation replayed offline from a condition bank (qd_damage.bank).

Usage:
    python -m qd_damage.offline --banks results/bank/me_seed0 ... --out results/offline/final

A trial on the damaged robot is one simulated episode of the chosen gait, taken from the first half of the
episodes stored for that condition; the quality of the recommended gait (the best one observed so far) is the mean
of the second half, which the method never sees. All methods share the same random episode draws (common random
numbers): the k-th trial of a given cell returns the same episode whatever the method.

Performance is expressed as a share of the progress of the best intact gait, u = (F - f0) / (F*_intact - f0),
with f0 the return of a standing robot; u is comparable across robots.

Methods (configs/adaptation.yaml, section `offline`):
- best_intact, random, top_k: as in the first campaign;
- ite: M-BOA (Cully et al. 2015), Gaussian process on the feet-contact descriptor; a cell can be tried again;
- ite_norep: same, but a cell is never tried twice;
- ite_joint: ite_norep on the joint-usage descriptor;
- robust_top_k: cells in decreasing order of intact performance + mean response to the simulated damages;
- aprol: one prior per simulated damage (and the intact one); after each trial, the prior that best explains the
  trials (Gaussian-process marginal likelihood) is used (after Kaushik et al. 2020);
- itesd: proposed. Prior = intact + mean simulated response; kernel = empirical covariance of the simulated
  responses between cells (+ a nugget), no repeated cell.
"""

import os

# Many tiny linear-algebra calls: one thread per process is much faster than a multi-threaded BLAS fighting over cores.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import zlib  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from qd_damage.config import CONFIG_DIR, load_yaml  # noqa: E402
from qd_damage.ite import matern52  # noqa: E402

REPO_ROOT = CONFIG_DIR.parent

TEST_GROUPS = ("id", "ood_jammed", "ood_inverted", "ood_sensor", "ood_slippery", "ood_heavy", "gap")


# ----------------------------------------------------------------------------------------------- data


def load_bank(bank_dir: Path) -> dict:
    """Bank of one repertoire, restricted to the filled cells and normalised to u."""
    bank_dir = Path(bank_dir)
    data = np.load(bank_dir / "bank.npz")
    info = json.loads((bank_dir / "bank.json").read_text())
    names, groups = list(data["names"]), list(data["groups"])
    filled = np.flatnonzero(np.isfinite(data["repertoire_fitness"]))
    fit = data["fitness"][:, filled]  # (conditions, n, episodes)
    f0 = float(data["f0"])
    intact = np.nanmean(fit[names.index("intact")], axis=-1)
    progress = float(intact.max() - f0)
    u = (fit - f0) / progress
    return {
        "name": bank_dir.name,
        "info": info,
        "names": names,
        "groups": groups,
        "u": u,
        "prior": u[names.index("intact")].mean(-1),
        "train": np.stack([np.nanmean(u[i], -1) for i, g in enumerate(groups) if g == "train"]),
        "descriptors": data["descriptors"][filled],
        "joint_usage": data["joint_usage"][filled],
        "filled": filled,
    }


class Trials:
    """Trials and ground truth of one test condition, with common random numbers across methods."""

    def __init__(self, u_cond: np.ndarray, rng: np.random.Generator, max_trials: int):
        n_eps = int(np.isfinite(u_cond[0]).sum())
        half = n_eps // 2
        self.pool = u_cond[:, :half]
        self.truth = u_cond[:, half:n_eps].mean(-1)
        # draws[i, k]: episode returned by the k-th trial of cell i
        self.draws = rng.integers(half, size=(u_cond.shape[0], max_trials))
        self.count = np.zeros(u_cond.shape[0], dtype=int)

    def __call__(self, i: int) -> float:
        k = self.count[i]
        self.count[i] += 1
        return float(self.pool[i, self.draws[i, k]])

    def reset(self):
        self.count[:] = 0


# ----------------------------------------------------------------------------------------------- kernels


class Matern:
    def __init__(self, x, lengthscale, variance):
        self.x, self.lengthscale, self.variance = np.asarray(x, float), lengthscale, variance

    def __call__(self, a, b):
        return self.variance * matern52(self.x[a], self.x[b], self.lengthscale)

    def diag(self):
        return np.full(len(self.x), self.variance)


class LowRank:
    """Empirical covariance of simulated responses: k(i, j) = sum_d U[i, d] U[j, d] + nugget [i == j]."""

    def __init__(self, u_mat, nugget):
        self.u, self.nugget = u_mat, nugget

    def __call__(self, a, b):
        a, b = np.atleast_1d(a), np.atleast_1d(b)
        return self.u[a] @ self.u[b].T + self.nugget * (a[:, None] == b[None, :])

    def diag(self):
        return (self.u**2).sum(1) + self.nugget


def empirical_kernel(train: np.ndarray, prior: np.ndarray, nugget: float, n_damages=None, rng=None):
    """(prior mean, kernel) learned from the simulated damages (rows of `train`, in u)."""
    if n_damages is not None:
        train = train[rng.choice(len(train), n_damages, replace=False)]
    residuals = train - prior  # (N, n)
    mean = residuals.mean(0)
    centred = (residuals - mean).T / np.sqrt(max(len(train) - 1, 1))
    return prior + mean, LowRank(centred, nugget)


# ----------------------------------------------------------------------------------------------- searches


def _posterior(prior, kernel, noise, tried, y):
    n = len(prior)
    k_tt = kernel(tried, tried) + noise * np.eye(len(tried))
    k_st = kernel(np.arange(n), tried)
    chol = np.linalg.cholesky(k_tt)
    a = np.linalg.solve(chol, k_st.T)
    mean = prior + a.T @ np.linalg.solve(chol, np.asarray(y) - prior[tried])
    var = np.maximum(kernel.diag() - (a**2).sum(0), 1e-12)
    return mean, var


def gp_search(prior, kernel, noise, trial, max_trials, kappa, repeat=False, alpha=0.9):
    """UCB search over the cells. Returns (tried cells, observations, stop trial or None)."""
    tried, y, stop = [], [], None
    mean, var = prior.copy(), kernel.diag()
    for t in range(max_trials):
        score = mean + kappa * np.sqrt(var)
        if not repeat and tried:
            score[tried] = -np.inf
        i = int(np.argmax(score))
        tried.append(i)
        y.append(trial(i))
        mean, var = _posterior(prior, kernel, noise, tried, y)
        if stop is None and max(y) >= alpha * mean.max():
            stop = t + 1
    return tried, y, stop


def _log_likelihood(kernel, noise, tried, residuals):
    k = kernel(tried, tried) + noise * np.eye(len(tried))
    chol = np.linalg.cholesky(k)
    a = np.linalg.solve(chol, residuals)
    return -0.5 * a @ a - np.log(np.diag(chol)).sum()


def aprol_search(priors, kernel, noise, trial, max_trials, kappa, alpha=0.9):
    """Prior selection (after APROL): the first prior is the intact one, used until a trial is observed."""
    tried, y, stop, best = [], [], None, 0
    mean, var = priors[0].copy(), kernel.diag()
    for t in range(max_trials):
        score = mean + kappa * np.sqrt(var)
        if tried:
            score[tried] = -np.inf
        i = int(np.argmax(score))
        tried.append(i)
        y.append(trial(i))
        obs = np.asarray(y)
        best = int(np.argmax([_log_likelihood(kernel, noise, tried, obs - p[tried]) for p in priors]))
        mean, var = _posterior(priors[best], kernel, noise, tried, y)
        if stop is None and max(y) >= alpha * mean.max():
            stop = t + 1
    return tried, y, stop


def order_search(order, trial, max_trials):
    tried = [int(i) for i in order[:max_trials]]
    return tried, [trial(i) for i in tried], None


# ----------------------------------------------------------------------------------------------- methods


def offline_config() -> dict:
    return load_yaml("adaptation")["offline"]


def method_runner(method: str, bank: dict, cfg: dict, robot: str):
    """Return run(trial, rng) -> (tried, observations, stop) for one method on one repertoire."""
    prior, max_trials = bank["prior"], cfg["max_trials"]
    gp = cfg["gp"][robot]
    if method == "best_intact":
        return lambda trial, rng: order_search([int(np.argmax(prior))], trial, 1)
    if method == "random":
        return lambda trial, rng: order_search(rng.permutation(len(prior)), trial, max_trials)
    if method == "top_k":
        order = np.argsort(-prior)
        return lambda trial, rng: order_search(order, trial, max_trials)
    if method in ("ite", "ite_norep"):
        kern = Matern(bank["descriptors"], gp["feet"]["lengthscale"], gp["feet"]["variance"])
        noise, kappa = gp["feet"]["noise"], cfg["kappa"]["ite"]
        repeat = method == "ite"
        return lambda trial, rng: gp_search(prior, kern, noise, trial, max_trials, kappa, repeat=repeat)
    if method == "ite_joint":
        kern = Matern(bank["joint_usage"], gp["joint"]["lengthscale"], gp["joint"]["variance"])
        noise, kappa = gp["joint"]["noise"], cfg["kappa"]["ite"]
        return lambda trial, rng: gp_search(prior, kern, noise, trial, max_trials, kappa)
    if method == "robust_top_k":
        order = np.argsort(-(prior + (bank["train"] - prior).mean(0)))
        return lambda trial, rng: order_search(order, trial, max_trials)
    if method == "aprol":
        kern = Matern(bank["descriptors"], gp["feet"]["lengthscale"], gp["feet"]["variance"])
        priors = [prior] + list(bank["train"])
        noise, kappa = gp["feet"]["noise"], cfg["kappa"]["ite"]
        return lambda trial, rng: aprol_search(priors, kern, noise, trial, max_trials, kappa)
    if method.startswith("itesd"):
        sd = dict(cfg["itesd"])
        # Ablations: itesd_N16 (16 simulated damages), itesd_nomean (intact prior), itesd_ucbk<kappa>.
        n_damages = None
        for part in method.split("_")[1:]:
            if part.startswith("N"):
                n_damages = int(part[1:])
            elif part == "nomean":
                sd["prior_mean"] = False
        mean_prior, kern = empirical_kernel(bank["train"], prior, sd["nugget"])

        def run(trial, rng):
            p, k = mean_prior, kern
            if n_damages is not None:  # a different subset of simulated damages for every repetition
                p, k = empirical_kernel(bank["train"], prior, sd["nugget"], n_damages, rng)
            if not sd["prior_mean"]:
                p = prior
            return gp_search(p, k, sd["noise"], trial, max_trials, sd["kappa"])

        return run
    raise ValueError(f"unknown method {method!r}")


def _seed(*parts) -> int:
    return zlib.crc32("/".join(map(str, parts)).encode())


def adapt_bank(bank_dir: Path, methods, n_repeats: int, cfg: dict, groups=TEST_GROUPS) -> list:
    """Rows (one per repertoire, condition, method, repetition, trial) for one bank."""
    bank = load_bank(bank_dir)
    robot = bank["info"]["robot"]
    runners = {m: method_runner(m, bank, cfg, robot) for m in methods}
    rows = []
    for c, (name, group) in enumerate(zip(bank["names"], bank["groups"], strict=True)):
        if group not in groups:
            continue
        for rep in range(n_repeats):
            trials = Trials(bank["u"][c], np.random.default_rng(_seed(bank["name"], name, rep)), cfg["max_trials"])
            for method, run in runners.items():
                trials.reset()
                tried, y, stop = run(trials, np.random.default_rng(_seed(bank["name"], name, rep, method)))
                best = np.maximum.accumulate(y)
                rec = [tried[int(np.argmax(y[: t + 1]))] for t in range(len(y))]
                for t in range(cfg["max_trials"]):
                    k = min(t, len(rec) - 1)  # one-trial methods keep their gait
                    rows.append(
                        (bank["name"], robot, bank["info"]["algo"], bank["info"]["seed"], name, group, method, rep,
                         t + 1, float(trials.truth[rec[k]]), float(best[k]), stop or 0,
                         len(set(tried[: k + 1])), float(trials.truth.max()))
                    )
    return rows


COLUMNS = ("run", "robot", "algo", "seed", "condition", "group", "method", "rep", "trial", "u_true", "u_observed",
           "stop_trial", "distinct_cells", "u_oracle")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--banks", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--groups", nargs="+", default=list(TEST_GROUPS))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "offline" / "final")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    cfg = offline_config()
    methods = args.methods or cfg["methods"]
    repeats = args.repeats or cfg["n_repeats"]
    args.out.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    with ProcessPoolExecutor(args.workers) as pool:
        futures = {b: pool.submit(adapt_bank, b, methods, repeats, cfg, tuple(args.groups)) for b in args.banks}
        for bank_dir, future in futures.items():
            df = pd.DataFrame(future.result(), columns=COLUMNS)
            df.to_csv(args.out / f"{Path(bank_dir).name}.csv.gz", index=False)
            print(f"[offline] {Path(bank_dir).name}: {len(df)} rows", flush=True)
    (args.out / "offline.json").write_text(json.dumps({"config": cfg, "methods": methods, "repeats": repeats}))


if __name__ == "__main__":
    main()

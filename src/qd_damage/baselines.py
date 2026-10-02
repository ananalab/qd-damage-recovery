"""Adaptation baselines, with the same interface as ITE (qd_damage.ite.run_ite).

- best-intact: replay the best gait of the intact robot (no adaptation, one trial);
- top-k: try cells in decreasing order of intact fitness (no Gaussian process);
- random: try cells drawn at random, without replacement.
"""

from typing import Callable

import numpy as np

from qd_damage.ite import TrialHistory


def _run_order(order, trial_fn: Callable[[int], float], max_trials: int) -> TrialHistory:
    history = TrialHistory()
    for i in order[:max_trials]:
        history.cells.append(int(i))
        history.observed.append(float(trial_fn(int(i))))
    return history


def run_best_intact(prior_fitness: np.ndarray, trial_fn: Callable[[int], float]) -> TrialHistory:
    return _run_order([int(np.argmax(prior_fitness))], trial_fn, 1)


def run_top_k(prior_fitness: np.ndarray, trial_fn: Callable[[int], float], max_trials: int) -> TrialHistory:
    return _run_order(np.argsort(-prior_fitness), trial_fn, max_trials)


def run_random(n_cells: int, trial_fn: Callable[[int], float], max_trials: int, rng: np.random.Generator) -> TrialHistory:
    return _run_order(rng.permutation(n_cells), trial_fn, max_trials)

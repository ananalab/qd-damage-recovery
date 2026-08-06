"""Adaptation baselines, with the same interface as ITE (qd_damage.ite.run_ite).

- best-intact: replay the best gait of the intact robot (no adaptation, one trial);
- top-k: try cells in decreasing order of intact fitness (no Gaussian process);
- random: try cells drawn at random, without replacement.
"""

from collections.abc import Callable

import numpy as np

from qd_damage.ite import TrialHistory



"""Damage adaptation: ITE and baselines, on real simulated episodes of the damaged robot.

Usage:
    python -m qd_damage.adaptation --runs results/repertoires/me_seed0 ...   (the oracle must exist)

For each repertoire, damage and method (ITE, top-k, random, best-intact), `n_repeats` repetitions (episodes are
noisy: random starts). Each trial is ONE episode on the damaged robot. The ITE prior (and the best-intact choice)
is the intact fitness re-evaluated by the oracle. The quality of the recommended gait (the best one observed) is
measured with the oracle mean over several episodes, which is less noisy than a single trial.

Outputs in results/adaptation/<run>/: trials.csv (one trial per row), summary.csv and adaptation.json.
Performances are percentages (F - F0) / (F_ref - F0), with F0 the return of a standing robot
(configs/adaptation.yaml) and F_ref the oracle under the damage or the intact oracle.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import jax
import numpy as np


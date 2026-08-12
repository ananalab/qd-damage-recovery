"""Alternative behaviour descriptor for ITE: how much each joint is used.

Usage:
    python -m qd_damage.joint_usage --runs results/repertoires/dcrlme_seed0 ...

The repertoires are organised by feet contact, but a damage acts on joints: a gait that hardly uses the front-right
hip should suffer little when that hip is paralysed. For every filled cell, this script plays the policy on the
intact robot and records the mean absolute action of each of the 8 actuators (in [0, 1], tanh outputs). ITE can
then model the effect of the damage on this 8D descriptor instead of the 4D feet-contact one (descriptor
ablation). Outputs in results/descriptors/<run>/joint_usage.npz (NaN for empty cells).
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from qd_damage.config import load_run_config
from qd_damage.oracle import evaluate_all, load_genotypes, make_evaluator
from qd_damage.repertoires import REPO_ROOT


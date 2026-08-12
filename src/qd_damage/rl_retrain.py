"""RL re-training (TD3) on the damaged robot, and a damage-robust TD3 policy.

Usage:
    python -m qd_damage.rl_retrain --scenario leg4_paralysed --variant finetune --seed 0 \
        --repertoire results/repertoires/dcrlme_seed0
    python -m qd_damage.rl_retrain --robust --seed 0

Variants trained on the damaged robot:
- scratch: TD3 from a random policy;
- finetune: TD3 from the best intact policy of the repertoire (best re-evaluated cell), with a new critic;
- finetune_warmup: same, but the actor is frozen during the first `warmup_episodes` so that the new critic learns
  before its gradient is used (these episodes are counted);
- finetune_pretrained: same, but the critic is first trained in simulation on the INTACT robot with the actor
  frozen (`pretrain_episodes`, not counted: no damaged robot is involved).

Performance (deterministic evaluation, same fitness as the repertoires) is recorded against the number of EPISODES
played on the damaged robot, which is what costs on a real robot. `n_envs` robots play in parallel; one round =
`episode_length` steps = `n_envs` episodes. Outputs in results/rl/<scenario>_<variant>_seed<seed>[_smoke]/.

The robust policy is trained in simulation with a random damage drawn for every episode (each joint independently
paralysed or weakened with probability `robust.p_damaged`), then evaluated without any adaptation on every damage
scenario: zero trials on the damaged robot. Outputs in results/rl_robust/seed<seed>[_smoke]/.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from qdax.baselines.td3 import TD3, TD3Config
from qdax.core.neuroevolution.buffers.buffer import ReplayBuffer, Transition

from qd_damage.config import load_run_config, load_yaml
from qd_damage.envs import DamageWrapper, damage_scale, get_scenario, load_damages, make_env
from qd_damage.oracle import make_evaluator
from qd_damage.repertoires import REPO_ROOT, git_commit, load_policy_params


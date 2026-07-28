"""Build a repertoire with MAP-Elites, PGA-ME or DCRL-ME, at equal budget.

Usage:
    python -m qd_damage.repertoires --algo me --seed 0 --profile smoke
    python -m qd_damage.repertoires --algo pgame --seed 0            # full run (GPU)

Everything that must be identical across algorithms (environment, network, repertoire, evaluations per
iteration) comes from common.yaml; only the emitter changes. The code follows the official QDax 0.5.0
notebooks (mapelites, pgame, dcrlme).

Outputs in <out>/<algo>_seed<seed>[_<profile>]/:
    repertoire.npz  flattened genotypes, fitnesses, descriptors, centroids
    metrics.csv     QD-score, coverage, max fitness and time, every log_period iterations
    run.json        config, seed, git commit, library versions, device
    actor.npz       DCRL-ME only: the descriptor-conditioned actor

A checkpoint is written every `budget.checkpoint_every` iterations and the run resumes from it automatically.
"""

import argparse
import csv
import functools
import json
import os
import pickle
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.emitters.dcrl_me_emitter import DCRLMEConfig, DCRLMEEmitter
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.pga_me_emitter import PGAMEConfig, PGAMEEmitter
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.buffers.buffer import DCRLTransition, QDTransition
from qdax.core.neuroevolution.networks.networks import MLP, MLPDC
from qdax.tasks.brax.v1 import reward_offset
from qdax.tasks.brax.v1.env_creators import scoring_function_brax_envs
from qdax.utils.metrics import default_qd_metrics

from qd_damage.config import ALGOS, load_run_config
from qd_damage.envs import descriptor_extractor, make_env


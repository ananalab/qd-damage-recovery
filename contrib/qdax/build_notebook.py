"""Build the example notebook proposed to QDax: ite_damage_recovery.ipynb (in English, like the QDax examples).

    python contrib/qdax/build_notebook.py && jupyter nbconvert --to notebook --execute contrib/qdax/ite_damage_recovery.ipynb \
        --output ite_damage_recovery_executed.ipynb
"""

import json
from pathlib import Path

CELLS = [
    (
        "markdown",
        """
# Damage recovery with Intelligent Trial & Error (ITE)

This notebook shows how to use a MAP-Elites repertoire built with QDax to let a robot **recover from damage
in a few trials**, with *Intelligent Trial & Error* (Cully et al., *Robots that can adapt like animals*,
Nature 2015).

1. Build a repertoire of Ant gaits with MAP-Elites (descriptor: feet contact).
2. Damage the robot: one leg is paralysed.
3. Use M-BOA (the Bayesian optimisation at the heart of ITE) to find a gait that still works, trying only a
   handful of policies on the damaged robot.

The repertoire gives, for each cell, the performance measured on the **intact** robot (the prior). After each
trial on the damaged robot, a Gaussian process corrects the predictions of all **similar** gaits (close in
descriptor space); the next trial maximises an upper confidence bound.
""",
    ),
    (
        "code",
        """
try:
    import qdax
except ImportError:
    !pip install qdax[cuda12]
""",
    ),
    (
        "code",
        """
import functools
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import qdax.tasks.brax.v1 as environments
from qdax.tasks.brax.v1.env_creators import scoring_function_brax_envs as scoring_function
from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.utils.metrics import default_qd_metrics
""",
    ),
    (
        "code",
        """
#@title Parameters
env_name = "ant_uni"
episode_length = 250
batch_size = 256
num_iterations = 300          # increase for better repertoires (e.g. 2000 on GPU)
num_centroids = 1024
policy_hidden_layer_sizes = (128, 128)
seed = 0
""",
    ),
    ("markdown", "## 1. Build a repertoire of gaits with MAP-Elites"),
    (
        "code",
        """
env = environments.create(env_name, episode_length=episode_length)
key = jax.random.key(seed)
policy_network = MLP(layer_sizes=policy_hidden_layer_sizes + (env.action_size,),
                     kernel_init=jax.nn.initializers.lecun_uniform(), final_activation=jnp.tanh)


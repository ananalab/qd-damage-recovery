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

def make_play_step_fn(env):
    def play_step_fn(env_state, policy_params, key):
        actions = policy_network.apply(policy_params, env_state.obs)
        next_state = env.step(env_state, actions)
        transition = QDTransition(
            obs=env_state.obs, next_obs=next_state.obs, rewards=next_state.reward, dones=next_state.done,
            actions=actions, truncations=next_state.info["truncation"],
            state_desc=env_state.info["state_descriptor"], next_state_desc=next_state.info["state_descriptor"])
        return next_state, policy_params, key, transition
    return play_step_fn

scoring_fn = functools.partial(scoring_function, episode_length=episode_length, play_reset_fn=env.reset,
                               play_step_fn=make_play_step_fn(env),
                               descriptor_extractor=environments.descriptor_extractor[env_name])
emitter = MixingEmitter(mutation_fn=None, variation_percentage=1.0, batch_size=batch_size,
                        variation_fn=functools.partial(isoline_variation, iso_sigma=0.005, line_sigma=0.05))
map_elites = MAPElites(scoring_function=scoring_fn, emitter=emitter,
                       metrics_function=functools.partial(default_qd_metrics,
                           qd_offset=environments.reward_offset[env_name] * episode_length))

key, k1, k2, k3 = jax.random.split(key, 4)
init_params = jax.vmap(policy_network.init)(jax.random.split(k1, batch_size), jnp.zeros((batch_size, env.observation_size)))
centroids = compute_cvt_centroids(num_descriptors=env.descriptor_length, num_init_cvt_samples=50000,
                                  num_centroids=num_centroids, minval=0.0, maxval=1.0, key=k2)
repertoire, emitter_state, _ = map_elites.init(init_params, centroids, k3)

@jax.jit
def run_block(carry):
    return jax.lax.scan(map_elites.scan_update, carry, (), length=10)

for i in range(num_iterations // 10):
    (repertoire, emitter_state, key), metrics = run_block((repertoire, emitter_state, key))
print("coverage %.1f %%, max fitness %.1f" % (metrics["coverage"][-1], metrics["max_fitness"][-1]))
""",
    ),
    (
        "markdown",
        """
## 2. Damage the robot

A damage multiplies the actions of some joints by a factor (0 = paralysed). For Ant, the torque of an actuator
is `action × strength`, so this is equivalent to weakening the actuator. Actions `(6, 7)` are the hip and ankle
of the front-right leg (order of the actuators in the Brax config).
""",
    ),
    (
        "code",
        """
class DamageWrapper:
    def __init__(self, env, action_scale):
        self.env, self.action_scale = env, jnp.asarray(action_scale)
    def reset(self, rng):
        return self.env.reset(rng)
    def step(self, state, action):
        return self.env.step(state, action * self.action_scale)
    def __getattr__(self, name):
        return getattr(self.env, name)

damaged_env = DamageWrapper(env, [1, 1, 1, 1, 1, 1, 0, 0])   # front-right leg paralysed
""",
    ),
    (
        "code",
        """
filled = np.flatnonzero(np.asarray(repertoire.fitnesses) > -np.inf)
params = jax.tree.map(lambda x: x[filled], repertoire.genotypes)
descriptors = np.asarray(repertoire.descriptors)[filled]

def make_evaluator(e):
    @jax.jit
    def evaluate(params_batch, key):
        f, _, _ = scoring_function(params_batch, key, episode_length, e.reset, make_play_step_fn(e),
                                   environments.descriptor_extractor[env_name])
        return f
    return evaluate

eval_intact, eval_damaged = make_evaluator(env), make_evaluator(damaged_env)

# Prior = intact performance, re-evaluated over 4 episodes (a single stored evaluation over-estimates elites).
key, *ks = jax.random.split(key, 5)
prior = np.mean([np.asarray(eval_intact(params, k)) for k in ks], axis=0)
# For this demo only: the "true" damaged performance of every cell (the oracle ITE does not have).
key, *ks = jax.random.split(key, 5)
oracle = np.mean([np.asarray(eval_damaged(params, k)) for k in ks], axis=0)

best_intact = int(np.argmax(prior))
print("best intact gait: %.1f intact, %.1f once damaged; best damaged gait in the repertoire: %.1f"
      % (prior[best_intact], oracle[best_intact], oracle.max()))
""",
    ),
    (
        "markdown",
        """
## 3. Intelligent Trial & Error (M-BOA)

Performances are normalised so that 0 ≈ a standing robot and 1 ≈ a good gait (≈ 15 m in 12.5 s); the GP
models the residual between observed and prior performance.
""",
    ),
    (
        "code",
        """
def matern52(a, b, lengthscale):
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)) / lengthscale
    return (1 + np.sqrt(5) * d + 5 / 3 * d**2) * np.exp(-np.sqrt(5) * d)

def ite(descriptors, prior, trial_fn, kappa=0.05, alpha=0.9, lengthscale=0.4,
        signal_variance=0.05, noise_variance=0.02, max_trials=20):
    tested, observed = [], []
    for t in range(max_trials):
        if tested:
            X = descriptors[tested]
            K = signal_variance * matern52(X, X, lengthscale) + noise_variance * np.eye(len(tested))
            k = signal_variance * matern52(descriptors, X, lengthscale)
            L = np.linalg.cholesky(K)
            a = np.linalg.solve(L.T, np.linalg.solve(L, np.array(observed) - prior[tested]))
            v = np.linalg.solve(L, k.T)
            mean, std = prior + k @ a, np.sqrt(np.maximum(signal_variance - (v**2).sum(0), 1e-12))
            if max(observed) >= alpha * mean.max():   # stopping criterion of Cully et al.
                break
        else:
            mean, std = prior, np.full(len(prior), np.sqrt(signal_variance))
        i = int(np.argmax(mean + kappa * std))         # UCB
        tested.append(i)
        observed.append(trial_fn(i))
    return tested, observed

f0 = 0.9948 * episode_length      # measured return of a standing robot (survival bonus, minus contact cost)
scale = 1.2 * episode_length
normalise = lambda f: (np.asarray(f) - f0) / scale

def trial_fn(i):
    global key
    key, k = jax.random.split(key)
    one = jax.tree.map(lambda x: x[i:i + 1], params)
    return float(normalise(eval_damaged(one, k)[0]))    # ONE episode on the damaged robot

tested, observed = ite(descriptors, normalise(prior), trial_fn)
best = tested[int(np.argmax(observed))]
print(f"ITE stopped after {len(tested)} trials")
print("recommended gait: %.2f   best intact gait once damaged: %.2f   oracle (tries everything): %.2f"
      % (normalise(oracle[best]), normalise(oracle[best_intact]), normalise(oracle.max())))
""",
    ),
    (
        "code",
        """
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(range(1, len(observed) + 1), observed, "o-", label="observed (one episode)")
ax.plot(range(1, len(observed) + 1), np.maximum.accumulate(observed), "k--", label="best so far")
ax.axhline(normalise(oracle[best_intact]), color="tab:red", ls=":", label="best intact gait, damaged")
ax.axhline(normalise(oracle.max()), color="tab:green", ls=":", label="oracle")
ax.set_xlabel("trial on the damaged robot"); ax.set_ylabel("normalised performance"); ax.legend(fontsize=8)
plt.show()
""",
    ),
    (
        "markdown",
        """
### Going further
- Compare with repertoires built by PGA-ME or DCRL-ME (see the corresponding QDax notebooks).
- Fit the GP hyper-parameters by marginal likelihood on held-out repertoires rather than using fixed values.
- Reference: A. Cully, J. Clune, D. Tarapore, J.-B. Mouret, *Robots that can adapt like animals*, Nature 521, 2015.
""",
    ),
]


def build(path: Path):
    cells = []
    for kind, src in CELLS:
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        cells.append(c)
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    build(Path(__file__).parent / "ite_damage_recovery.ipynb")

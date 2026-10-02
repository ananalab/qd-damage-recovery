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

REPO_ROOT = Path(__file__).resolve().parents[2]


def make_policy_network(common: dict, action_size: int) -> MLP:
    """Same MLP for the three algorithms: hidden layers from common.yaml, tanh output."""
    return MLP(
        layer_sizes=tuple(common["policy"]["hidden_layer_sizes"]) + (action_size,),
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )


def make_play_step_fn(env, policy_network, algo: str):
    """One simulation step. DCRL-ME also stores descriptor fields in its transitions."""

    def play_step_fn(env_state, policy_params, key):
        actions = policy_network.apply(policy_params, env_state.obs)
        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)
        common_fields = dict(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            truncations=next_state.info["truncation"],
            actions=actions,
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
        )
        if algo == "dcrlme":
            nan_desc = jnp.full((env.descriptor_length,), jnp.nan)
            transition = DCRLTransition(**common_fields, desc=nan_desc, desc_prime=nan_desc)
        else:
            transition = QDTransition(**common_fields)
        return next_state, policy_params, key, transition

    return play_step_fn


def make_emitter(algo: str, algo_cfg: dict, common: dict, env, policy_network):
    """The only part that differs between algorithms. Each emitter produces `batch_size` offspring."""
    batch_size = common["budget"]["batch_size"]
    em = algo_cfg["emitter"]
    variation_fn = functools.partial(
        isoline_variation, iso_sigma=em["iso_sigma"], line_sigma=em["line_sigma"]
    )

    if algo == "me":
        return MixingEmitter(
            mutation_fn=None,
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=batch_size,
        )

    if algo == "pgame":
        td3 = algo_cfg["td3"]
        config = PGAMEConfig(
            env_batch_size=batch_size,
            batch_size=td3["transitions_batch_size"],
            proportion_mutation_ga=em["proportion_mutation_ga"],
            critic_hidden_layer_size=tuple(td3["critic_hidden_layer_size"]),
            critic_learning_rate=td3["critic_learning_rate"],
            greedy_learning_rate=td3["greedy_learning_rate"],
            policy_learning_rate=td3["policy_learning_rate"],
            noise_clip=td3["noise_clip"],
            policy_noise=td3["policy_noise"],
            discount=td3["discount"],
            reward_scaling=td3["reward_scaling"],
            replay_buffer_size=td3["replay_buffer_size"],
            soft_tau_update=td3["soft_tau_update"],
            num_critic_training_steps=td3["num_critic_training_steps"],
            num_pg_training_steps=td3["num_pg_training_steps"],
            policy_delay=td3["policy_delay"],
        )
        return PGAMEEmitter(
            config=config, policy_network=policy_network, env=env, variation_fn=variation_fn
        )

    if algo == "dcrlme":
        dcrl = algo_cfg["dcrl"]
        ga_batch_size = int(round(em["ga_fraction"] * batch_size))
        dcrl_batch_size = int(round(em["dcrl_fraction"] * batch_size))
        config = DCRLMEConfig(
            ga_batch_size=ga_batch_size,
            dcrl_batch_size=dcrl_batch_size,
            ai_batch_size=batch_size - ga_batch_size - dcrl_batch_size,
            lengthscale=dcrl["lengthscale"],
            critic_hidden_layer_size=tuple(dcrl["critic_hidden_layer_size"]),
            num_critic_training_steps=dcrl["num_critic_training_steps"],
            num_pg_training_steps=dcrl["num_pg_training_steps"],
            batch_size=dcrl["transitions_batch_size"],
            replay_buffer_size=dcrl["replay_buffer_size"],
            discount=dcrl["discount"],
            reward_scaling=dcrl["reward_scaling"],
            critic_learning_rate=dcrl["critic_learning_rate"],
            actor_learning_rate=dcrl["actor_learning_rate"],
            policy_learning_rate=dcrl["policy_learning_rate"],
            noise_clip=dcrl["noise_clip"],
            policy_noise=dcrl["policy_noise"],
            soft_tau_update=dcrl["soft_tau_update"],
            policy_delay=dcrl["policy_delay"],
        )
        actor_dc_network = make_dc_actor_network(common, env.action_size)
        return DCRLMEEmitter(
            config=config,
            policy_network=policy_network,
            actor_network=actor_dc_network,
            env=env,
            variation_fn=variation_fn,
        )

    raise ValueError(f"unknown algo {algo!r}")


def setup(algo: str, profile=None):
    """Build environment, network and MAP-Elites for one algorithm; also return the config used."""
    cfg = load_run_config(algo, profile)
    common = cfg["common"]
    env = make_env(common)
    policy_network = make_policy_network(common, env.action_size)

    scoring_fn = functools.partial(
        scoring_function_brax_envs,
        episode_length=common["env"]["episode_length"],
        play_reset_fn=jax.jit(env.reset),
        play_step_fn=make_play_step_fn(env, policy_network, algo),
        descriptor_extractor=descriptor_extractor(common),
    )
    # With positive_rewards the fitness is already >= 0, so the QD-score needs no offset.
    qd_offset = (
        0.0
        if common["env"]["positive_rewards"]
        else reward_offset[common["env"]["name"]] * common["env"]["episode_length"]
    )
    emitter = make_emitter(algo, cfg["algo"], common, env, policy_network)
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=emitter,
        metrics_function=functools.partial(default_qd_metrics, qd_offset=qd_offset),
    )
    return cfg, env, policy_network, map_elites


def git_commit() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True)
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True)
        return commit.strip() + ("-dirty" if dirty.strip() else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Outside a git checkout (e.g. on Kaggle) the launcher passes the commit.
        return os.environ.get("QD_DAMAGE_COMMIT", "unknown")


class _Key:
    """Picklable wrapper for a JAX random key (typed keys do not go through np.asarray)."""

    def __init__(self, data):
        self.data = data


def _to_host(tree):
    def leaf(x):
        if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
            return _Key(np.asarray(jax.random.key_data(x)))
        return np.asarray(x) if isinstance(x, jax.Array) else x
    return jax.tree.map(leaf, tree)


def _to_device(tree):
    def leaf(x):
        if isinstance(x, _Key):
            return jax.random.wrap_key_data(jnp.asarray(x.data))
        return jnp.asarray(x) if isinstance(x, np.ndarray) else x
    return jax.tree.map(leaf, tree, is_leaf=lambda x: isinstance(x, _Key))


def _save_checkpoint(path: Path, carry, iteration: int, rows: list, elapsed_s: float):
    """Write to a temporary file, then rename it, so that an interruption never leaves a corrupt checkpoint."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"carry": _to_host(carry), "iteration": iteration, "rows": rows, "elapsed_s": elapsed_s}, f)
    os.replace(tmp, path)


def build(
    algo: str,
    seed: int,
    profile=None,
    out_root: Path = REPO_ROOT / "results" / "repertoires",
    stop_after=None,
):
    """Build a repertoire, write repertoire.npz, metrics.csv and run.json, and return the output folder.

    Resumes from <out>/checkpoint.pkl when it exists (interrupted session).
    `stop_after` (iterations) stops the run on purpose after a checkpoint; used by the resume test.
    """
    cfg, env, policy_network, map_elites = setup(algo, profile)
    common = cfg["common"]
    budget, rep = common["budget"], common["repertoire"]
    if budget["num_iterations"] is None:
        raise ValueError("budget.num_iterations is not set in common.yaml")
    log_period = budget["log_period"]
    checkpoint_every = budget.get("checkpoint_every") or budget["num_iterations"]
    if budget["num_iterations"] % log_period or checkpoint_every % log_period:
        raise ValueError("budget.num_iterations and budget.checkpoint_every must be multiples of budget.log_period")

    run_name = f"{algo}_seed{seed}" + (f"_{profile}" if profile else "")
    out_dir = Path(out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=budget["batch_size"])
    init_params = jax.vmap(policy_network.init)(
        keys, jnp.zeros((budget["batch_size"], env.observation_size))
    )

    key, subkey = jax.random.split(key)
    centroids = compute_cvt_centroids(
        num_descriptors=env.descriptor_length,
        num_init_cvt_samples=rep["num_init_cvt_samples"],
        num_centroids=rep["num_centroids"],
        minval=rep["min_descriptor"],
        maxval=rep["max_descriptor"],
        key=subkey,
    )

    t0 = time.time()
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = map_elites.init(init_params, centroids, subkey)
    rows = [{
        "iteration": 0,
        "evaluations": budget["batch_size"],
        **{k: float(init_metrics[k]) for k in ("qd_score", "coverage", "max_fitness")},
        "time_s": time.time() - t0,
    }]

    num_loops = budget["num_iterations"] // log_period
    checkpoint_path = out_dir / "checkpoint.pkl"
    start_loop, previous_elapsed = 0, 0.0
    if checkpoint_path.exists():
        # The init above only provides the structure and the compiled functions; it is replaced here.
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        repertoire, emitter_state, key = _to_device(ckpt["carry"])
        rows, previous_elapsed = ckpt["rows"], ckpt["elapsed_s"]
        start_loop = ckpt["iteration"] // log_period
        print(f"[{run_name}] resuming from iteration {ckpt['iteration']}", flush=True)
    else:
        print(f"[{run_name}] init : {rows[-1]}", flush=True)

    # Compiled once. Calling jax.lax.scan(map_elites.scan_update, ...) directly in the loop recompiles at
    # every block, because `map_elites.scan_update` is a new bound-method object at each access.
    @jax.jit
    def run_block(carry):
        return jax.lax.scan(map_elites.scan_update, carry, (), length=log_period)

    for i in range(start_loop, num_loops):
        start = time.time()
        (repertoire, emitter_state, key), metrics = run_block((repertoire, emitter_state, key))
        jax.block_until_ready(repertoire.fitnesses)
        iteration = log_period * (i + 1)
        rows.append({
            "iteration": iteration,
            "evaluations": budget["batch_size"] * (iteration + 1),  # +1: the initial batch
            **{k: float(metrics[k][-1]) for k in ("qd_score", "coverage", "max_fitness")},
            "time_s": time.time() - start,  # the first block includes JIT compilation
        })
        print(f"[{run_name}] {rows[-1]}", flush=True)
        if iteration % checkpoint_every == 0 and iteration < budget["num_iterations"]:
            _save_checkpoint(
                checkpoint_path, (repertoire, emitter_state, key), iteration, rows,
                previous_elapsed + time.time() - t0,
            )
            if stop_after is not None and iteration >= stop_after:
                print(f"[{run_name}] stopped on purpose at iteration {iteration}", flush=True)
                return out_dir
    total_time = previous_elapsed + time.time() - t0

    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    flat_genotypes = jax.vmap(lambda p: ravel_pytree(p)[0])(repertoire.genotypes)
    np.savez_compressed(
        out_dir / "repertoire.npz",
        genotypes=np.asarray(flat_genotypes),
        fitnesses=np.asarray(repertoire.fitnesses).reshape(-1),
        descriptors=np.asarray(repertoire.descriptors),
        centroids=np.asarray(repertoire.centroids),
    )

    if algo == "dcrlme":
        # Descriptor-conditioned actor pi(a | obs, d), used by the continuous-ITE experiment.
        dcrl_state = next(st for st in emitter_state.emitter_states if hasattr(st, "actor_params"))
        np.savez_compressed(out_dir / "actor.npz", params=np.asarray(ravel_pytree(dcrl_state.actor_params)[0]))

    run_info = {
        "algo": algo,
        "seed": seed,
        "profile": profile,
        "config": cfg,
        "git_commit": git_commit(),
        "versions": {p: version(p) for p in ("qdax", "jax", "jaxlib", "brax", "flax")},
        "devices": [f"{d} ({d.device_kind})" for d in jax.devices()],
        "total_time_s": total_time,
        "total_evaluations": rows[-1]["evaluations"],
    }
    with open(out_dir / "run.json", "w") as f:
        json.dump(run_info, f, indent=2)
    checkpoint_path.unlink(missing_ok=True)  # the run is complete: the (large) checkpoint is no longer needed
    print(f"[{run_name}] done in {total_time:.0f} s -> {out_dir}", flush=True)
    return out_dir


def make_dc_actor_network(common: dict, action_size: int) -> MLPDC:
    return MLPDC(
        layer_sizes=tuple(common["policy"]["hidden_layer_sizes"]) + (action_size,),
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )


def load_dc_actor(run_dir: Path, common: dict, env):
    """(network, parameters) of the conditioned actor saved by a DCRL-ME run (actor.npz)."""
    actor = make_dc_actor_network(common, env.action_size)
    template = actor.init(jax.random.key(0), obs=jnp.zeros((env.observation_size,)),
                          desc=jnp.zeros((env.descriptor_length,)))
    _, unravel = ravel_pytree(template)
    return actor, unravel(jnp.asarray(np.load(Path(run_dir) / "actor.npz")["params"]))


def load_policy_params(run_dir: Path, index: int, policy_network, observation_size: int):
    """Rebuild the policy parameters of cell `index` from repertoire.npz."""
    data = np.load(Path(run_dir) / "repertoire.npz")
    template = policy_network.init(jax.random.key(0), jnp.zeros((observation_size,)))
    _, unravel = ravel_pytree(template)
    return unravel(jnp.asarray(data["genotypes"][index]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--algo", choices=ALGOS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", choices=["smoke", "local", "demo"], default=None)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "repertoires")
    args = parser.parse_args()
    build(args.algo, args.seed, args.profile, args.out)


if __name__ == "__main__":
    main()

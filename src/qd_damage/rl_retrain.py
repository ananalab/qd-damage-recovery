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

VARIANTS = ("scratch", "finetune", "finetune_warmup", "finetune_pretrained")


class _BatchedEnv:
    """n robots in parallel (vmap). QDax's AutoResetWrapper restarts every finished episode."""

    def __init__(self, env):
        self.env = env
        self.reset = jax.vmap(env.reset)
        self.step = jax.vmap(env.step)


class RandomDamageWrapper:
    """A new random damage at every episode: each joint is independently scaled by 0 or 0.5 with probability p.

    The damage lives in the state info, so it survives QDax's automatic resets and is redrawn when an episode ends.
    """

    def __init__(self, env, p_damaged: float):
        self.env = env
        self.p_damaged = p_damaged

    def _draw(self, key):
        k1, k2 = jax.random.split(key)
        damaged = jax.random.bernoulli(k1, self.p_damaged, (self.env.action_size,))
        level = jnp.where(jax.random.bernoulli(k2, 0.5, (self.env.action_size,)), 0.0, 0.5)
        return jnp.where(damaged, level, 1.0)

    def reset(self, rng):
        state = self.env.reset(rng)
        key, subkey = jax.random.split(jax.random.fold_in(rng, 1))
        state.info["damage_scale"] = self._draw(subkey)
        state.info["damage_key"] = key
        return state

    def step(self, state, action):
        scale, key = state.info["damage_scale"], state.info["damage_key"]
        state = self.env.step(state, action * scale)
        key, subkey = jax.random.split(key)
        state.info["damage_scale"] = jnp.where(state.done, self._draw(subkey), scale)
        state.info["damage_key"] = key
        return state

    def __getattr__(self, name):
        return getattr(self.env, name)


def load_rl_config(smoke: bool) -> dict:
    cfg = load_yaml("rl")
    smoke_cfg = cfg.pop("smoke")
    if smoke:
        for key, value in smoke_cfg.items():
            if isinstance(value, dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    return cfg


def best_intact_cell(repertoire: Path, oracle_root: Path) -> tuple:
    """(cell, source) of the best intact policy: on the re-evaluated intact fitness when the oracle exists,
    as for ITE and the best-intact baseline, otherwise on the fitness stored in the repertoire."""
    oracle_file = Path(oracle_root) / Path(repertoire).name / "oracle.npz"
    if oracle_file.exists():
        oracle = np.load(oracle_file)
        intact = oracle["fitness"][list(oracle["scenarios"]).index("intact")].mean(-1)
        return int(np.nanargmax(intact)), "oracle"
    return int(np.nanargmax(np.load(Path(repertoire) / "repertoire.npz")["fitnesses"])), "repertoire"


def make_td3(cfg: dict, common: dict, action_size: int) -> TD3:
    t = cfg["td3"]
    return TD3(
        TD3Config(
            episode_length=common["env"]["episode_length"],
            batch_size=t["batch_size"],
            policy_delay=t["policy_delay"],
            soft_tau_update=t["soft_tau_update"],
            expl_noise=t["expl_noise"],
            critic_hidden_layer_size=tuple(t["critic_hidden_layer_size"]),
            policy_hidden_layer_size=tuple(common["policy"]["hidden_layer_sizes"]),  # same network as the repertoires
            critic_learning_rate=t["critic_learning_rate"],
            policy_learning_rate=t["policy_learning_rate"],
            discount=t["discount"],
            noise_clip=t["noise_clip"],
            policy_noise=t["policy_noise"],
            reward_scaling=t["reward_scaling"],
        ),
        action_size=action_size,
    )


def make_train_round(td3: TD3, venv: _BatchedEnv, cfg: dict, episode_length: int, freeze_actor: bool = False):
    """Compiled round: episode_length steps for n_envs robots, updates_per_step gradient steps per step.

    With freeze_actor, only the critic learns: the policy, its target and its optimiser state are kept.
    """

    def one_update(carry, _):
        state, buffer = carry
        new_state, buffer, _ = td3.update(state, buffer)
        if freeze_actor:
            new_state = new_state.replace(
                policy_params=state.policy_params,
                target_policy_params=state.target_policy_params,
                policy_optimizer_state=state.policy_optimizer_state,
            )
        return (new_state, buffer), None

    def one_step(carry, _):
        env_state, state, buffer = carry
        env_state, state, transitions = td3.play_step_fn(env_state, state, venv)
        buffer = buffer.insert(transitions)
        (state, buffer), _ = jax.lax.scan(one_update, (state, buffer), (), length=cfg["updates_per_step"])
        return (env_state, state, buffer), None

    @jax.jit
    def train_round(env_state, state, buffer):
        (env_state, state, buffer), _ = jax.lax.scan(one_step, (env_state, state, buffer), (), length=episode_length)
        return env_state, state, buffer

    return train_round


def new_buffer(cfg: dict, env) -> ReplayBuffer:
    return ReplayBuffer.init(
        buffer_size=cfg["buffer_size"], transition=Transition.init_dummy(env.observation_size, env.action_size)
    )


def write_curve(out_dir: Path, rows: list, info: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "curve.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "rl.json").write_text(
        json.dumps(
            {**info, "code_commit": git_commit(), "devices": [f"{d} ({d.device_kind})" for d in jax.devices()]},
            indent=2,
        )
    )


def retrain(
    scenario: str,
    variant: str,
    seed: int,
    repertoire=None,
    smoke: bool = False,
    out_root: Path = REPO_ROOT / "results" / "rl",
    oracle_root: Path = REPO_ROOT / "results" / "oracle",
):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
    if variant != "scratch" and repertoire is None:
        raise ValueError(f"--repertoire is required for {variant}")
    cfg = load_rl_config(smoke)
    common = load_run_config("me", "smoke" if smoke else None)["common"]
    episode_length = common["env"]["episode_length"]
    damages = load_damages()
    scale = damage_scale(get_scenario(scenario, damages), damages)
    env = DamageWrapper(make_env(common), scale)
    venv = _BatchedEnv(env)
    td3 = make_td3(cfg, common, env.action_size)

    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    state = td3.init(subkey, env.action_size, env.observation_size)

    _, policy_network, evaluate = make_evaluator(common)
    init_cell = init_source = None
    if variant != "scratch":
        # Same architecture (MLP (128, 128), tanh output): the repertoire weights load as they are.
        init_cell, init_source = best_intact_cell(repertoire, oracle_root)
        params = load_policy_params(repertoire, init_cell, policy_network, env.observation_size)
        assert jax.tree.structure(params) == jax.tree.structure(state.policy_params)
        state = state.replace(policy_params=params, target_policy_params=params)

    pretrain_time = 0.0
    if variant == "finetune_pretrained":
        # Critic trained on the intact robot, in simulation, with the actor frozen; then a fresh buffer.
        t_pre = time.time()
        intact = _BatchedEnv(make_env(common))
        pre_round = make_train_round(td3, intact, cfg, episode_length, freeze_actor=True)
        pre_key = jax.random.fold_in(jax.random.key(seed), 1)
        pre_state = intact.reset(jax.random.split(pre_key, cfg["n_envs"]))
        pre_buffer = new_buffer(cfg, env)
        for _ in range(cfg["pretrain_episodes"] // cfg["n_envs"]):
            pre_state, state, pre_buffer = pre_round(pre_state, state, pre_buffer)
        pretrain_time = time.time() - t_pre

    buffer = new_buffer(cfg, env)
    key, subkey = jax.random.split(key)
    env_state = venv.reset(jax.random.split(subkey, cfg["n_envs"]))
    train_round = make_train_round(td3, venv, cfg, episode_length)
    frozen_round = make_train_round(td3, venv, cfg, episode_length, freeze_actor=True)
    warmup = cfg["warmup_episodes"] if variant == "finetune_warmup" else 0

    def evaluate_policy(policy_params, key):
        batch = jax.tree.map(lambda x: jnp.repeat(x[None], cfg["eval_episodes"], axis=0), policy_params)
        fitnesses, _ = evaluate(batch, key, jnp.asarray(scale))
        return float(fitnesses.mean())

    rows, t0, episodes = [], time.time(), 0
    key, subkey = jax.random.split(key)
    rows.append({"episodes": 0, "eval_fitness": evaluate_policy(state.policy_params, subkey), "time_s": 0.0})
    while episodes < cfg["max_episodes"]:
        step_fn = frozen_round if episodes < warmup else train_round
        env_state, state, buffer = step_fn(env_state, state, buffer)
        episodes += cfg["n_envs"]
        key, subkey = jax.random.split(key)
        rows.append(
            {
                "episodes": episodes,
                "eval_fitness": evaluate_policy(state.policy_params, subkey),
                "time_s": time.time() - t0,
            }
        )
        if len(rows) % 10 == 0 or episodes >= cfg["max_episodes"]:
            print(
                f"[rl {scenario} {variant} s{seed}] {episodes} episodes: {rows[-1]['eval_fitness']:.1f} "
                f"({rows[-1]['time_s']:.0f} s)",
                flush=True,
            )

    out_dir = Path(out_root) / (f"{scenario}_{variant}_seed{seed}" + ("_smoke" if smoke else ""))
    write_curve(
        out_dir,
        rows,
        {
            "scenario": scenario,
            "variant": variant,
            "seed": seed,
            "repertoire": str(repertoire),
            "init_cell": init_cell,
            "init_cell_source": init_source,
            "warmup_episodes": warmup,
            "pretrain_time_s": pretrain_time,
            "config": cfg,
            "time_s": time.time() - t0,
        },
    )
    return out_dir


def train_robust(seed: int, smoke: bool = False, out_root: Path = REPO_ROOT / "results" / "rl_robust"):
    """TD3 from scratch with a random damage at every episode, then zero-shot evaluation on every scenario."""
    cfg = load_rl_config(smoke)
    common = load_run_config("me", "smoke" if smoke else None)["common"]
    episode_length = common["env"]["episode_length"]
    env = RandomDamageWrapper(make_env(common), cfg["robust"]["p_damaged"])
    venv = _BatchedEnv(env)
    td3 = make_td3(cfg, common, env.action_size)
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    state = td3.init(subkey, env.action_size, env.observation_size)
    buffer = new_buffer(cfg, env)
    key, subkey = jax.random.split(key)
    env_state = venv.reset(jax.random.split(subkey, cfg["n_envs"]))
    train_round = make_train_round(td3, venv, cfg, episode_length)

    damages = load_damages()
    _, _, evaluate = make_evaluator(common)

    def evaluate_on(policy_params, scenario, key):
        batch = jax.tree.map(lambda x: jnp.repeat(x[None], cfg["eval_episodes"], axis=0), policy_params)
        fitnesses, _ = evaluate(batch, key, jnp.asarray(damage_scale(scenario, damages)))
        return float(fitnesses.mean())

    rows, t0, episodes, every = [], time.time(), 0, cfg["robust"]["eval_every"]
    while episodes < cfg["robust"]["episodes"]:
        env_state, state, buffer = train_round(env_state, state, buffer)
        episodes += cfg["n_envs"]
        if episodes % every < cfg["n_envs"] or episodes >= cfg["robust"]["episodes"]:
            key, subkey = jax.random.split(key)
            fits = [
                evaluate_on(state.policy_params, s, k)
                for s, k in zip(damages["scenarios"], jax.random.split(subkey, len(damages["scenarios"])), strict=False)
            ]
            rows.append(
                {
                    "episodes": episodes,
                    "intact": fits[0],
                    "damaged_mean": float(np.mean(fits[1:])),
                    "time_s": time.time() - t0,
                }
            )
            print(
                f"[robust s{seed}] {episodes} episodes: intact {fits[0]:.1f}, damaged mean "
                f"{rows[-1]['damaged_mean']:.1f} ({rows[-1]['time_s']:.0f} s)",
                flush=True,
            )

    # Final zero-shot evaluation, more episodes per scenario.
    key, subkey = jax.random.split(key)
    final = []
    for s, k in zip(damages["scenarios"], jax.random.split(subkey, len(damages["scenarios"])), strict=False):
        batch = jax.tree.map(
            lambda x: jnp.repeat(x[None], cfg["robust"]["final_eval_episodes"], axis=0), state.policy_params
        )
        fits, _ = evaluate(batch, k, jnp.asarray(damage_scale(s, damages)))
        final.append(
            {
                "seed": seed,
                "scenario": s["name"],
                "fitness": float(fits.mean()),
                "std": float(fits.std()),
                "n_episodes": cfg["robust"]["final_eval_episodes"],
            }
        )
    out_dir = Path(out_root) / (f"seed{seed}" + ("_smoke" if smoke else ""))
    write_curve(out_dir, rows, {"variant": "robust", "seed": seed, "config": cfg, "time_s": time.time() - t0})
    with open(out_dir / "zero_shot.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(final[0]))
        writer.writeheader()
        writer.writerows(final)
    np.savez_compressed(out_dir / "policy.npz", params=np.asarray(ravel_pytree(state.policy_params)[0]))
    return out_dir


def evaluate_robust(robust_dir: Path, physics=None, n_episodes: int = 32, seed: int = 7) -> Path:
    """Zero-shot evaluation of a saved robust policy on every damage, optionally on perturbed physics."""
    robust_dir = Path(robust_dir)
    info = json.loads((robust_dir / "rl.json").read_text())
    common = load_run_config("me")["common"]
    env, policy_network, evaluate = make_evaluator(common, physics=physics)
    template = policy_network.init(jax.random.key(0), jnp.zeros((env.observation_size,)))
    _, unravel = ravel_pytree(template)
    params = unravel(jnp.asarray(np.load(robust_dir / "policy.npz")["params"]))
    damages = load_damages()
    rows = []
    keys = jax.random.split(jax.random.key(seed), len(damages["scenarios"]))
    for s, k in zip(damages["scenarios"], keys, strict=True):
        batch = jax.tree.map(lambda x: jnp.repeat(x[None], n_episodes, axis=0), params)
        fits, _ = evaluate(batch, k, jnp.asarray(damage_scale(s, damages)))
        rows.append(
            {
                "seed": info["seed"],
                "scenario": s["name"],
                "fitness": float(fits.mean()),
                "std": float(fits.std()),
                "n_episodes": n_episodes,
            }
        )
    out = robust_dir / ("zero_shot_gap.csv" if physics else "zero_shot_check.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenario")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--robust", action="store_true", help="train the damage-robust policy instead")
    parser.add_argument(
        "--evaluate-robust",
        type=Path,
        default=None,
        help="evaluate a saved robust policy (results/rl_robust/seed<s>) on every damage",
    )
    parser.add_argument("--reality-gap", action="store_true", help="with --evaluate-robust: perturbed physics")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repertoire", type=Path, default=None, help="required for the finetune variants")
    parser.add_argument("--oracle-root", type=Path, default=REPO_ROOT / "results" / "oracle")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.evaluate_robust:
        physics = load_yaml("adaptation")["reality_gap"] if args.reality_gap else None
        print(evaluate_robust(args.evaluate_robust, physics))
    elif args.robust:
        print(train_robust(args.seed, args.smoke, args.out or REPO_ROOT / "results" / "rl_robust"))
    else:
        if not (args.scenario and args.variant):
            parser.error("--scenario and --variant are required unless --robust")
        print(
            retrain(
                args.scenario,
                args.variant,
                args.seed,
                args.repertoire,
                args.smoke,
                args.out or REPO_ROOT / "results" / "rl",
                args.oracle_root,
            )
        )


if __name__ == "__main__":
    main()

# Development notes

Decisions, checks and pitfalls, in the order they came up. Final numbers are in `results/analysis/summary.md`
and `report/generated/numbers.tex`.

## Setup

- Prior work: Chalumeau et al. (ICLR 2023) already damaged Ant-uni (weakened or paralysed leg) with MAP-Elites,
  PGA-ME and skill-discovery RL, but selected the gait with an oracle (each skill evaluated 100 times). No ITE,
  no trial count, no DCRL-ME, no RL re-training. See `docs/protocol.md`.
- QDax 0.5.0 ships two environment modules, `qdax.tasks.brax.v1` and `.v2`. At tag v0.5.0 the three official
  notebooks (mapelites, pgame, dcrlme) import `v1`, so `v1` is used everywhere.
- The notebooks do not share a default environment (`walker2d_uni` for PGA-ME, `ant_omni` for DCRL-ME):
  `ant_uni` is forced through `configs/common.yaml`.
- Without pins, pip installs jax 0.11 and brax 0.14, where `brax.v1` no longer exists and QDax 0.5.0 breaks.
  The pins come from QDax's `requirements.txt` at v0.5.0 (jax/jaxlib 0.4.28, brax 0.10.4, flax 0.8.5,
  chex 0.1.86, optax 0.1.9, numpy 1.26.4, tensorflow-probability 0.24.0), plus mujoco / mujoco-mjx 3.1.5
  (recent mujoco breaks with jax 0.4.28). Python 3.11 or 3.12 only. Exact versions: `requirements-lock.txt`.

## A common configuration for the three algorithms

Differences between the official notebooks:

| | MAP-Elites | PGA-ME | DCRL-ME |
|---|---|---|---|
| default env | walker2d_uni | walker2d_uni | ant_omni |
| episode_length | 100 | 250 | 250 |
| policy MLP | (64, 64) | (256, 256) | (128, 128) |
| evaluations / iteration | 100 | 100 | 256 (128 GA + 64 DCRL + 64 actor) |
| reward | raw | raw | shifted by reward_offset, clipped at 0 |
| iso / line sigma, CVT | 0.005 / 0.05, 1024 cells | same | same |


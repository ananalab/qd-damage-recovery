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

Shared choices (`configs/common.yaml`):
- `ant_uni`, 250-step episodes (the value of both RL notebooks), descriptor in [0, 1]^4.
- MLP (128, 128) for all three: the DCRL-ME size, between the two others.
- 256 evaluations per iteration for all three (MAP-Elites: 256 mutations; PGA-ME: 128 GA + 128 gradient;
  DCRL-ME: 128 / 64 / 64), and the same number of iterations, hence the same number of evaluations.
- Shifted and clipped reward for all three: DCRL-ME needs it, and fitness then means the same thing everywhere.
  The QD-score needs no offset since fitness is already >= 0.
- Algorithm-specific hyper-parameters are those of the notebooks (`configs/algos/*.yaml`).

Implementation notes:
- QDax 0.5.0 has no `save()` for repertoires: genotypes are flattened with `ravel_pytree`.
- Calling `jax.lax.scan(map_elites.scan_update, ...)` inside the loop recompiles at every block (a new bound
  method each time). The block is compiled once with `jax.jit`.
- QDax environments have an `AutoResetWrapper`: the last state of an episode is already reset. Distances are
  computed on the last state before the reset.
- On GPU the initial repertoires differ slightly between algorithms with the same seed (initial QD-score 109k /
  113k / 115k), while they were identical on CPU. Most likely different floating-point rounding in the compiled
  programs, amplified by the physics. For the same reason, re-running an adaptation on CPU does not reproduce the
  GPU trial sequence exactly; the reported results are the Kaggle (GPU) ones.

## Checkpoints and the full runs

- A checkpoint (repertoire, emitter state including the RL replay buffer, random key) is written every 250
  iterations to a temporary file then renamed; runs resume automatically. Typed JAX keys do not go through
  `np.asarray`, so they are converted with `jax.random.key_data` / `wrap_key_data`.
- Test: a run interrupted and resumed gives exactly the same repertoire (genotypes and fitnesses) as a straight
  run, for the three algorithms.
- Kaggle pitfalls: brax 0.10.4 requires `pytinyrenderer`, which does not install on Python 3.12, so the exact
  local versions are installed with `--no-deps` and rendering stays local; macOS `tar` adds `._*` metadata files
  that break pip on Linux (`COPYFILE_DISABLE=1`, `--no-mac-metadata`, and a check in `kaggle/push_code.sh`);
  Kaggle sometimes extracts the code archive itself, so the kernels handle both cases; a `kaggle kernels status`
  call can hang, hence the per-call timeout in `kaggle/wait_kernel.py`.

## Damages and oracle


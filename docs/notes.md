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

Leg mapping, checked in three ways:
1. Geometry at reset (the robot walks towards +x, +y on its left): foot Body 4 (x+, y+) is front left,
   Body 7 (x-, y+) hind left, Body 10 (x-, y-) hind right, Body 13 (x+, y-) front right.
2. Measurement: a single action set to 1 for 4 steps moves the foot relative to its thigh the most for the
   expected leg (actions 2k, 2k+1 drive leg k+1). Absolute foot displacement did not work: the whole body moves.
3. Video: `media/checks/leg_mapping/`.

So leg 1 = actions (0, 1) front left, 2 = (2, 3) hind left, 3 = (4, 5) hind right, 4 = (6, 7) front right, and
the descriptor lists the feet in the same order.

Two findings that changed the analysis:
1. The zero of performance. A standing robot scores 1058.7 (64 episodes), not 810: Brax adds +1 per step for
   survival on top of the 3.24 offset. All percentages use F0 = 4.2348 x episode length. With the wrong zero,
   MAP-Elites looked ~97 % resilient although its gaits barely move.
2. Elites are over-estimated. The stored fitness comes from one noisy episode and MAP-Elites keeps the lucky
   evaluations. Re-evaluated on 4 episodes, the "best" cell of the demo MAP-Elites repertoire does worse than a
   standing robot; the correlation between stored and re-evaluated fitness is 0.88. The ITE prior, the
   best-intact baseline and the best intact cell of the oracle summary therefore use the fitness re-evaluated by
   the oracle (legitimate: offline, before the damage).

## ITE

- M-BOA in numpy: Gaussian process on the residuals, Matern 5/2 kernel, UCB, stop at alpha. Baselines:
  best-intact, top-k, random. Every trial is a real simulated episode on the damaged robot; the recommended gait
  is scored with the oracle mean.
- Synthetic test problem (two families of gaits, the damage destroys the better one): ITE recovers 97 % of the
  optimum in 2 trials while top-k stays at 0.19 after 15 trials. A first version of this problem was wrong
  (the damage improved some regions) and was fixed.
- Performance scale p = (F - F0) / 300 (about 15 m in 12.5 s). On this fixed, shared scale the variance of one
  episode (measured on the oracle) is ~0.001, the value of Cully et al.
- Gaussian-process hyper-parameters: with Cully's values (length-scale 0.4, variance 1, noise 0.001), ITE stopped
  after 3.4 trials and plateaued. The variogram of the residuals on the development set has a variance of
  0.03 to 0.06 (not 1) and a large nugget (neighbouring cells differ). Maximum marginal likelihood
  (`qd_damage.calibrate`, 3 demo repertoires x 12 damages x 300 cells) gives length-scale 0.4, signal variance
  0.05, noise 0.02 (log-likelihood +4059 against -1280 for Cully's values). kappa from 0.05 to 2 made little
  difference, so kappa = 0.05 and alpha = 0.9 (Cully).
- Development / test split: all these choices were made on the 200-iteration demo repertoires; the final
  campaign uses the 9 full repertoires, which were not used for any choice.
- The oracle is itself optimistic (maximum of noisy means over ~700 to 900 cells), so "% of oracle"
  under-estimates every method.

## RL re-training

- QDax TD3, 16 robots in parallel, one gradient step per transition, deterministic evaluation after every round
  of 16 episodes, up to 2000 episodes. Variants: from scratch, or from the best intact DCRL-ME policy (same MLP,
  so the weights load as they are; new critic). The x axis counts episodes on the damaged robot.
- The reported fine-tuning runs started from the elite with the highest stored fitness, because the oracle ran in
  parallel on Kaggle. For seed 1 it is also the best re-evaluated cell; for seed 0 it is cell 724 instead of
  cell 43, whose re-evaluated intact fitness is 1556 against 1578 (1.4 % lower). The code now waits for the
  oracle and uses the re-evaluated best cell; `rl.json` records the starting cell.

## Analysis and statistics

- Every number of the reports is a LaTeX macro written by `qd_damage.analysis.report`; tables are generated too.
- The absolute performance p was added because percentages "of the intact gait" hide that the best intact
  MAP-Elites gait barely moves.
- The five repetitions of an adaptation only differ by episode noise. They are averaged before any test, so the
  unit is a (repertoire, damage) pair (36 per algorithm). A first version tested the 180 repetitions as if they
  were independent, which inflated significance. Units from the same seed share a repertoire, so p-values remain
  indicative.
- ITE against top-k is tested with paired Wilcoxon tests on these units. A first reading of pooled medians
  suggested that ITE clearly helps on DCRL-ME repertoires; the paired test only supports a small advantage after
  3 trials (DCRL-ME p = 0.02; all repertoires p = 0.009), which is gone after 5 and 10 trials.

## Visuals

- `qd_damage.viz`: repertoire figures, videos and offline 3D pages. The Brax v0.1.0 viewer and three.js r135 are
  bundled in `src/qd_damage/assets/` (Apache 2.0 / MIT licences included) and inlined in each page. Headless
  Firefox does not render WebGL, so the 3D pages were checked by hand in a browser.
- `qd_damage.viz_ite`: the three-phase video (intact / damaged, same gait / damaged, after ITE), close-ups, 3D,
  ITE trial figure, repertoire map before and after damage, still frames for the PDF. Each video shows the median
  of 8 episodes, since a single episode is too noisy to be representative. The ITE run shown is the median
  repetition in number of trials.
- On the demo DCRL-ME repertoire with leg 4 paralysed, ITE barely helps (3.5 m against 3.8 m).

## Continuous ITE

- DCRL-ME saves its descriptor-conditioned actor (`actor.npz`). `qd_damage.ite_continuous` lets ITE search 2048
  gaits pi(.|d) for targets d spread over [0,1]^4, on the same run as grid ITE. Three dedicated DCRL-ME runs were
  needed because the first campaign did not save the actor.
- Clear negative result: the actor's gaits are much weaker than the elites, both for ITE and for the oracle.

## Open points

- The QDax example notebook: make the demo more telling (stronger damage or longer repertoire) before any PR.
- Descriptors closer to what a damage changes; a pre-trained critic for TD3 fine-tuning.

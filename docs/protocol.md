# Question and protocol

## Question

A four-legged robot (the Brax Ant) loses the use of a leg. What lets it walk again with the fewest trials?

| Approach | Idea |
|---|---|
| MAP-Elites (evolution only) | Build a repertoire of gaits, then try the most promising ones |
| PGA-ME / DCRL-ME (evolution + RL) | Same, but the repertoire is built with the help of a TD3 critic |
| RL re-training | Train the gait again, directly on the damaged robot |

We measure the number of trials on the damaged robot (what is expensive on a real robot) and the performance
recovered. Adaptation uses Intelligent Trial & Error (ITE, Cully et al., Nature 2015): Bayesian optimisation
with a Gaussian process over the repertoire.

## Prior work and what this project adds

Chalumeau et al. (ICLR 2023) tested Ant-uni with a weakened or paralysed leg (MAP-Elites, PGA-ME and
skill-discovery RL). Their post-damage selection is an oracle: every skill is evaluated 100 times on the damaged
robot and the best one is kept. They found that PGA-ME adapts better than MAP-Elites, and that QD methods are
more robust to large perturbations.

This project adds:
1. the actual cost in trials, with ITE, depending on whether the repertoire was built with or without RL;
2. DCRL-ME (Faldor et al., ACM TELO 2024);
3. a comparison with RL re-training, counted in episodes on the damaged robot;
4. as a side experiment, ITE in the continuous descriptor space of the DCRL-ME conditioned actor, instead of the
   repertoire cells.

## Protocol

- Environment: `ant_uni` (descriptor: fraction of time each foot touches the ground), same Brax version for all.
- Policies: the same MLP for the three algorithms.
- Budget: the same number of evaluations; 3 seeds per algorithm.
- Damages: 12 scenarios fixed in advance in `configs/damages.yaml`.
- ITE: kappa, alpha and the maximum number of trials fixed in advance (`configs/adaptation.yaml`); Gaussian-process
  hyper-parameters fitted on development repertoires (200-iteration demo runs) only, then frozen; 5 repetitions
  per (repertoire, damage) pair.
- RL: counted in episodes on the damaged robot, not in compute time.
- Statistics: repetitions are averaged first; the unit is a (repertoire, damage) pair. Mann-Whitney tests between
  algorithms, paired Wilcoxon tests between adaptation methods.

## Rules followed

1. Every full run is first tested with `--profile smoke` on CPU.
2. The three algorithms use exactly the same environment, policy network and total number of evaluations;
   only the emitter differs. Tests check it (`tests/test_setup.py`, `tests/test_repertoires.py`).
3. Every number in the README and the reports comes from `results/` (through `report/generated/`) or from a
   `data/summary.json` in `media/`.
4. Long scripts can resume from regular checkpoints.
5. Every run records its config, seed, git commit, QDax / JAX / Brax versions and device.

# Second campaign: what limits few-trial adaptation from a repertoire?

Written on 13/10/2026, before any of the runs below. The hypotheses, the conditions and the analysis are fixed here;
the hyper-parameters of the adaptation methods are chosen on the development repertoires only (200-iteration runs),
then frozen in `configs/adaptation.yaml` before the final repertoires are analysed.

## Why

Since QD algorithms use RL critics (PGA-ME, DCRL-ME), adaptation to damage is mostly evaluated with an oracle that
tries every gait of the repertoire; the number of trials a real robot would need is no longer measured. The first
campaign found that, with such repertoires, ITE is barely better than trying gaits in decreasing order of intact
performance. A pilot replay of the first campaign's data (13/10/2026) suggests why: ITE keeps re-testing the same few
gaits, and the intact ranking is a poor prior, while the response of the repertoire to damages simulated before the
real one gives both a better prior and a better similarity between gaits than the behaviour descriptor.

## Hypotheses

- H1: ITE as published plateaus because it re-tests cells; forbidding repeated cells improves the recommended gait
  after 10 and 20 trials.
- H2: ITE-SD (prior and kernel learned from simulated damages, `qd_damage.offline`) beats ITE and top-k after 3, 5
  and 10 trials on in-family damages that were never simulated as such.
- H3: on out-of-family damages and with a reality gap, ITE-SD is at least as good as ITE.
- H4: the gain of a model over top-k grows when the intact ranking is a poor prior (low rank correlation between
  intact and damaged performance) and when the damage response lies in the span of the simulated responses.
- H5: H1 and H2 hold on the three robots and for the three repertoire algorithms.

## Design

- Robots: Ant (5 seeds per algorithm: 0 to 2 from the first campaign, 3 and 4 new), Walker2d and HalfCheetah
  (3 seeds per algorithm), same budget and hyper-parameters as the first campaign (`configs/robots.yaml`).
- Condition bank (`configs/conditions.yaml`, `qd_damage.bank`): every gait under the intact robot, 64 simulated
  training damages drawn from the family of the damage-robust TD3 policy, in-family named damages, out-of-family
  damages (jammed motor, inverted motor, lost joint sensors, slippery foot, heavy foot) and two levels of reality gap.
  Training damages equal to a test damage are redrawn.
- Adaptation replayed offline (`qd_damage.offline`): trials drawn from the first half of the episodes of a test
  condition, recommended gait measured on the second half; 20 repetitions; common random numbers across methods.
  Validated against real episodes on the repertoires of the first campaign.
- Baselines: best intact, random, top-k, ITE, ITE without repeats, ITE on joint usage, top-k on the simulated prior,
  APROL-like prior selection, and the damage-robust TD3 policy trained on the same damage family (zero trials).
- Statistics: unit = (repertoire, condition), repetitions averaged; IQM with stratified bootstrap intervals; paired
  Wilcoxon tests and mixed models with a random intercept per repertoire; Holm correction over H1 to H3.

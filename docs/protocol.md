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

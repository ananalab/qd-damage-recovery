# Compute budget

Measured values only. Used to choose the number of evaluations so that one run fits in 1 to 2 hours on a T4.

## Time per iteration

- 2000 iterations x 256 = 512,256 evaluations per run, the same for the three algorithms.
- Estimated time per run on a T4: MAP-Elites ~7 min, PGA-ME ~57 min, DCRL-ME ~86 min (plus ~1 min of
  installation and compilation).
- 3 algorithms x 3 seeds = 9 runs, ~7.5 GPU hours. A Kaggle session has two T4s, so two runs go in parallel.


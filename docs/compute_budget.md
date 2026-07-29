# Compute budget

Measured values only. Used to choose the number of evaluations so that one run fits in 1 to 2 hours on a T4.

## Time per iteration

256 evaluations per iteration and 250-step episodes for every algorithm. Steady-state time = mean over the
10-iteration blocks after the first one (the first block includes JIT compilation).

The T4 timings come from the `time_s` column of `results/repertoires/*_seed0_demo/metrics.csv`.

DCRL-ME is the slowest algorithm, so it sets the budget.

- 2000 iterations x 256 = 512,256 evaluations per run, the same for the three algorithms.
- Estimated time per run on a T4: MAP-Elites ~7 min, PGA-ME ~57 min, DCRL-ME ~86 min (plus ~1 min of
  installation and compilation).
- 3 algorithms x 3 seeds = 9 runs, ~7.5 GPU hours. A Kaggle session has two T4s, so two runs go in parallel.

## Actual durations of the 9 full runs (Kaggle, Tesla T4)


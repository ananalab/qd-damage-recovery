# Compute budget

Measured values only. Used to choose the number of evaluations so that one run fits in 1 to 2 hours on a T4.

## Time per iteration

256 evaluations per iteration and 250-step episodes for every algorithm. Steady-state time = mean over the
10-iteration blocks after the first one (the first block includes JIT compilation).

| Algorithm | Machine | Time / iteration (steady state) | First block of 10 (with compilation) | 200-iteration run |
|---|---|---|---|---|
| MAP-Elites | Kaggle Tesla T4 (demo profile) | 0.22 s | 16.1 s | 69 s |
| PGA-ME | Kaggle Tesla T4 (demo profile) | 1.70 s | 38.8 s | 376 s |
| DCRL-ME | Kaggle Tesla T4 (demo profile, 3000 critic steps) | 2.56 s | 50.7 s | 555 s |
| MAP-Elites | Mac arm64 CPU (local profile) | 1.45 s | 18.3 s | 306 s |
| PGA-ME | Mac arm64 CPU (local profile) | ~30 s (CPU shared with video rendering) | | 6119 s |
| DCRL-ME | Mac arm64 CPU (local profile, 300 critic steps) | ~20 s | | stopped at iteration 30 |

The T4 timings come from the `time_s` column of `results/repertoires/*_seed0_demo/metrics.csv`.

## Chosen budget

DCRL-ME is the slowest algorithm, so it sets the budget.

- 2000 iterations x 256 = 512,256 evaluations per run, the same for the three algorithms.
- Estimated time per run on a T4: MAP-Elites ~7 min, PGA-ME ~57 min, DCRL-ME ~86 min (plus ~1 min of
  installation and compilation).
- 3 algorithms x 3 seeds = 9 runs, ~7.5 GPU hours. A Kaggle session has two T4s, so two runs go in parallel.

## Actual durations of the 9 full runs (Kaggle, Tesla T4)

| Algorithm | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| MAP-Elites | 7.6 min | 7.7 min | 7.7 min |
| PGA-ME | 52.4 min | 53.1 min | 52.3 min |
| DCRL-ME | 81.7 min | 81.5 min | 83.2 min |

About 4 hours for the 9 runs with two in parallel (`results/repertoires/*/run.json`, logs in
`results/kaggle_logs/full/`). The evaluation campaign (oracle, adaptation and 16 TD3 runs) took about 2.5 hours
more, and the continuous-ITE experiment about 97 minutes per seed.

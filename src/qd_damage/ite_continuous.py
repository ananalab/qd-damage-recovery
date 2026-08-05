"""Continuous ITE: search the descriptor space with the descriptor-conditioned actor of DCRL-ME.

Usage:
    python -m qd_damage.ite_continuous --runs results/bonus/repertoires/dcrlme_seed0 ...

DCRL-ME learns an actor pi(a | observation, d): "walk with feet-contact profile d". Instead of searching the
~1000 cells of the repertoire, ITE searches N gaits produced by the actor for N target descriptors spread over
[0,1]^4, i.e. gaits "between the cells". Everything else is identical to ITE on the grid (re-evaluated prior,
Gaussian process, damages, number of repetitions), so both can be compared on the same run.

Outputs in results/bonus/continuous/<run>/: candidates.npz, trials.csv, summary.csv and continuous.json.
"""


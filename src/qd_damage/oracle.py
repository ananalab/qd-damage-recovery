"""Oracle: evaluate every policy of a repertoire under every damage.

Usage:
    python -m qd_damage.oracle --runs results/repertoires/me_seed0 results/repertoires/pgame_seed0 ...

For each damage of configs/damages.yaml and each filled cell, `n_episodes` episodes (different random starts)
are played and averaged. This gives:
- the oracle performance: the best policy of the repertoire under this damage. It is the ceiling of what ITE
  can find, but the oracle tries everything (about 1000 x n_episodes episodes per damage);
- the performance of the best intact policy replayed on the damaged robot (no adaptation);
- the resilience of the repertoire: (damaged oracle - F0) / (intact oracle - F0), F0 = standing robot.


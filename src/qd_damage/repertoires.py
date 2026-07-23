"""Build a repertoire with MAP-Elites, PGA-ME or DCRL-ME, at equal budget.

Usage:
    python -m qd_damage.repertoires --algo me --seed 0 --profile smoke
    python -m qd_damage.repertoires --algo pgame --seed 0            # full run (GPU)

Everything that must be identical across algorithms (environment, network, repertoire, evaluations per
iteration) comes from common.yaml; only the emitter changes. The code follows the official QDax 0.5.0
notebooks (mapelites, pgame, dcrlme).


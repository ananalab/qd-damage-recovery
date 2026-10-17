# Few-trial damage recovery with Quality-Diversity repertoires

A simulated four-legged robot loses the use of a leg. How many trials does it need to walk again, and does it
matter whether its repertoire of gaits was built with or without reinforcement learning?

![Intact, damaged, and after Intelligent Trial & Error](media/final/ite_main/videos/three_phases_follow.gif)

*Brax Ant, DCRL-ME repertoire, front-right leg paralysed. Each panel plays the median of 8 episodes and shows its
distance in 12.5 s; averaged over the 8 episodes: 30.6 m intact, 2.8 m damaged with the same gait, 9.4 m with the
gait found by ITE in 6 trials.*

## What I did

- Built gait repertoires at equal budget with **MAP-Elites**, **PGA-ME** and **DCRL-ME** (3 seeds each,
  512,256 episodes per run) using JAX, QDax and Brax on Kaggle GPUs.
- Damaged the robot in 12 ways (paralysed or weakened legs, ankles, two legs).
- Adapted with **Intelligent Trial & Error** (Cully et al., *Nature* 2015), compared with simple baselines, with
  **TD3 re-training** and with a **damage-robust TD3 policy**, counting every episode played on the damaged robot.
- Checked the conclusions against oracle bias, dependence between units (mixed models), the choice of descriptor
  and a simulated reality gap.

## Results

| | MAP-Elites | PGA-ME | DCRL-ME |
|---|---|---|---|
| Coverage | 72 % | 77 % | **84 %** |
| Best gait left after damage (v, 1 ≈ 15 m) | 0.23 | **1.84** | 1.14 |
| Gait found by ITE (v) | 0.11 | **1.47** | 0.59 |

- Repertoires built with RL are at least as resilient as MAP-Elites in relative terms and much better in absolute
  terms (rank tests and mixed models, oracle bias checked on fresh episodes).
- ITE stops after a median of **5 trials** and recovers 33 % of the intact progress (18 % without adaptation).
  Simply trying gaits by decreasing intact performance does almost as well: ITE's small edge after 3 trials is
  gone after 5. This holds with a descriptor that models the damage three times better, and with a gap between
  the simulator and the robot. The quality of the repertoire matters more than the adaptation algorithm.
- Re-training with TD3 takes hundreds of episodes on the damaged robot. Fine-tuning the intact policy collapses
  unless its critic is warmed up or pre-trained; then it matches ITE in 8 of 8 runs, after a median of 400 to 450 episodes.
- A TD3 policy trained in simulation on random damages matches ITE on the best repertoires **without any trial**
  (better on 6 of 12 damages), but only for damages it was trained on, and it fails when two legs are paralysed.
- Letting ITE search the gaits of DCRL-ME's descriptor-conditioned actor instead of the repertoire is clearly worse.

Full report: [`report/report.pdf`](report/report.pdf). Every number in it is generated from `results/`.

## Run it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,viz]"          # CPU; on a GPU: pip install -e ".[cuda12]"
pytest -m "not slow"

python -m qd_damage.repertoires --algo dcrlme --seed 0    # add --profile smoke for a quick CPU run
python -m qd_damage.oracle      --runs results/repertoires/dcrlme_seed0
python -m qd_damage.adaptation  --runs results/repertoires/dcrlme_seed0
python -m qd_damage.rl_retrain  --scenario leg4_paralysed --variant scratch --seed 0
python -m qd_damage.analysis.report
```

## Layout

```
src/qd_damage/   environment and damages, repertoires, oracle, ITE and baselines, TD3, analysis, visuals
configs/         shared settings, per-algorithm hyper-parameters, damages, ITE, RL
kaggle/          GPU scripts used for the campaign
results/         metrics and summaries of every run (repertoires themselves are not versioned)
media/           videos, interactive 3D pages, figures
report/          report and its generated tables
docs/            protocol, development notes, compute budget
contrib/qdax/    standalone ITE example notebook for QDax
```

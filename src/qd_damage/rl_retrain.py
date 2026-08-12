"""RL re-training (TD3) on the damaged robot, and a damage-robust TD3 policy.

Usage:
    python -m qd_damage.rl_retrain --scenario leg4_paralysed --variant finetune --seed 0 \
        --repertoire results/repertoires/dcrlme_seed0
    python -m qd_damage.rl_retrain --robust --seed 0

Variants trained on the damaged robot:
- scratch: TD3 from a random policy;
- finetune: TD3 from the best intact policy of the repertoire (best re-evaluated cell), with a new critic;
- finetune_warmup: same, but the actor is frozen during the first `warmup_episodes` so that the new critic learns
  before its gradient is used (these episodes are counted);
- finetune_pretrained: same, but the critic is first trained in simulation on the INTACT robot with the actor
  frozen (`pretrain_episodes`, not counted: no damaged robot is involved).


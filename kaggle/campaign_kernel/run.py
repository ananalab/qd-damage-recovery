"""Kaggle (2 x T4): evaluation campaign on the 9 repertoires (outputs of the qd-damage-recovery-full kernel).

- for each repertoire: oracle (every policy x every damage), then adaptation (ITE, top-k, random, best-intact;
  5 repetitions; real simulated episodes);
- TD3 re-training on 4 damages x 2 variants (from scratch / from the best intact DCRL-ME policy) x 2 seeds.
  Fine-tuning runs after the oracle of its repertoire, so that it starts from the best RE-EVALUATED intact policy.
Two jobs at a time (one per GPU). Outputs in /kaggle/working/outputs/{oracle,adaptation,rl}/.
"""


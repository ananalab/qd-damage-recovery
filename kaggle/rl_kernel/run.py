"""Kaggle (2 x T4): stronger RL baselines.

- damage-robust TD3 policy (random damage at every episode), 2 seeds, evaluated zero-shot on every damage;
- TD3 fine-tuning seed 0 re-run from the best RE-EVALUATED intact DCRL-ME policy (the first campaign used the elite
  with the highest stored fitness because the oracle ran in parallel);
- fine-tuning with a critic warm-up (actor frozen at first) and with a critic pre-trained on the intact robot,
  4 damages x 2 seeds;
- reality gap: oracle of every repertoire on perturbed physics, then ITE, top-k and best-intact with the prior from
  the nominal simulator and the trials on the perturbed robot.
Inputs: the repertoires (qd-damage-recovery-full) and their oracle (qd-damage-recovery-campaign).
Outputs in /kaggle/working/outputs/{rl,rl_robust,oracle_gap,adaptation_gap}/.
"""

import glob
import os
import shutil
import sys
import tarfile


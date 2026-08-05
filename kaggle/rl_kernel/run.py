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

CODE = "/kaggle/working/code"
# The dataset holds the code archive, which Kaggle sometimes extracts by itself: handle both cases.
archives = glob.glob("/kaggle/input/**/qd_damage_code.tar.gz", recursive=True)
if archives:
    with tarfile.open(archives[0]) as tar:
        tar.extractall(CODE)
else:
    src = os.path.dirname(glob.glob("/kaggle/input/**/CODE_COMMIT", recursive=True)[0])
    shutil.copytree(src, CODE)  # copy: /kaggle/input is read-only and pip -e needs to write
sys.path.insert(0, f"{CODE}/kaggle")
import kernel_utils as ku  # noqa: E402


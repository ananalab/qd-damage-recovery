"""Kaggle (2 x T4): continuous ITE with the descriptor-conditioned actor of DCRL-ME.

For each seed: a full DCRL-ME run that saves its actor (actor.npz), then the oracle and the grid adaptation
(ITE and baselines), then continuous ITE, all on the SAME run.
Outputs in /kaggle/working/outputs/{repertoires,oracle,adaptation,continuous}/.
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

env = ku.install(CODE)
OUT, py = "/kaggle/working/outputs", sys.executable
jobs = []
for seed in [0, 1, 2]:
    run = f"{OUT}/repertoires/dcrlme_seed{seed}"
    jobs.append((f"bonus_dcrlme_seed{seed}", ["bash", "-c", " && ".join([
        f"{py} -m qd_damage.repertoires --algo dcrlme --seed {seed} --out {OUT}/repertoires",
        f"{py} -m qd_damage.oracle --runs {run} --out {OUT}/oracle",
        f"{py} -m qd_damage.adaptation --runs {run} --oracle-root {OUT}/oracle --out {OUT}/adaptation",
        f"{py} -m qd_damage.ite_continuous --runs {run} --out {OUT}/continuous",
    ])]))
ku.run_jobs(jobs, CODE, env, log_dir="/kaggle/working/logs")

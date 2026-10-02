"""Kaggle (2 x T4): the 9 full repertoires, 3 algorithms x 3 seeds, 2000 iterations each.

Two runs at a time (one per GPU), longest first (DCRL-ME, then PGA-ME, then MAP-Elites).
Outputs in /kaggle/working/outputs/<algo>_seed<seed>/, logs in /kaggle/working/logs/.
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
jobs = [
    (f"{algo}_seed{seed}",
     [sys.executable, "-m", "qd_damage.repertoires", "--algo", algo, "--seed", str(seed),
      "--out", "/kaggle/working/outputs"])
    for algo in ["dcrlme", "pgame", "me"]  # longest first
    for seed in [0, 1, 2]
]
ku.run_jobs(jobs, CODE, env, log_dir="/kaggle/working/logs")

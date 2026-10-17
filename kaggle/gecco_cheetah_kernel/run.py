"""Kaggle (2 x T4): final repertoires of the second campaign, same budget as the first one (2000 iterations x 256).

HalfCheetah seeds 0 to 2, for the three algorithms (Ant and Walker2d: gecco_reps_kernel).
Outputs in /kaggle/working/outputs/<run>/.
"""

import glob
import os
import shutil
import sys
import tarfile

CODE = "/kaggle/working/code"
archives = glob.glob("/kaggle/input/**/qd_damage_code.tar.gz", recursive=True)
if archives:
    with tarfile.open(archives[0]) as tar:
        tar.extractall(CODE)
else:
    src = os.path.dirname(glob.glob("/kaggle/input/**/CODE_COMMIT", recursive=True)[0])
    shutil.copytree(src, CODE)
sys.path.insert(0, f"{CODE}/kaggle")
import kernel_utils as ku  # noqa: E402

ROBOTS_SEEDS = {"halfcheetah": (0, 1, 2)}
env = ku.install(CODE)
jobs = [
    (
        f"{robot}_{algo}_seed{seed}",
        [
            ku.PYTHON,
            "-m",
            "qd_damage.repertoires",
            "--algo",
            algo,
            "--seed",
            str(seed),
            "--robot",
            robot,
            "--out",
            "/kaggle/working/outputs",
        ],
    )
    for algo in ("dcrlme", "pgame", "me")  # longest first
    for robot, seeds in ROBOTS_SEEDS.items()
    for seed in seeds
]
ku.run_jobs(jobs, CODE, env, log_dir="/kaggle/working/logs")

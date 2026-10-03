"""Kaggle (T4): the 3 demo repertoires (demo profile, 200 iterations, seed 0), one after the other.

Used to time the algorithms on GPU and as the development set for tuning ITE.
Outputs in /kaggle/working/outputs/<algo>_seed0_demo/.
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
for algo in ["me", "pgame", "dcrlme"]:
    ku.sh(
        [
            ku.PYTHON,
            "-m",
            "qd_damage.repertoires",
            "--algo",
            algo,
            "--seed",
            "0",
            "--profile",
            "demo",
            "--out",
            "/kaggle/working/outputs",
        ],
        cwd=CODE,
        env=env,
    )
print("ALL_DONE", flush=True)

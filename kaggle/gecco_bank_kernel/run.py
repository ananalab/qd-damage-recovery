"""Kaggle (2 x T4): condition banks with proprioceptive signatures, and closed- vs open-loop play, for every
repertoire of the second campaign (final and development).

Inputs: the repertoires of the first campaign (Ant seeds 0 to 2 and development runs) and of the second one
(gecco-dev, gecco-reps, gecco-cheetah). Two jobs at a time (one per GPU).
Outputs in /kaggle/working/outputs/{bank,feedback}/<run>/.
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

env = ku.install(CODE)
OUT, py = "/kaggle/working/outputs", ku.PYTHON
runs = {}
for path in glob.glob("/kaggle/input/**/repertoire.npz", recursive=True):
    name = os.path.basename(os.path.dirname(path))
    if not name.endswith("_smoke") and name not in runs:
        runs[name] = os.path.dirname(path)
final = sorted(n for n in runs if not n.endswith("_demo"))
dev = sorted(n for n in runs if n.endswith("_demo"))
print(len(final), "final:", final, flush=True)
print(len(dev), "development:", dev, flush=True)

jobs = []
for name in final + dev:
    steps = [f"{py} -m qd_damage.bank --signatures --runs {runs[name]} --out {OUT}/bank"]
    if name in final:
        episodes = 4 if "_" not in name.split("_seed")[0] else 2  # Ant runs have no robot prefix
        steps.append(f"{py} -m qd_damage.feedback --runs {runs[name]} --out {OUT}/feedback --episodes {episodes}")
    jobs.append((name, ["bash", "-c", " && ".join(steps)]))
ku.run_jobs(jobs, CODE, env, log_dir="/kaggle/working/logs")

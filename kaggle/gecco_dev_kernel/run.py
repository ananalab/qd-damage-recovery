"""Kaggle (2 x T4): first session of the second campaign.

- development repertoires of Walker2d and HalfCheetah (demo profile, 200 iterations, seed 0), then their banks;
- damage-robust TD3 policies of Walker2d and HalfCheetah (seeds 0 and 1), then their banks;
- banks of the 9 Ant repertoires of the first campaign and of the 3 Ant development repertoires;
- banks of the two Ant robust policies of the first campaign.
Two jobs at a time (one per GPU), longest first. Outputs in /kaggle/working/outputs/.
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


def find_dirs(pattern):
    return sorted({os.path.dirname(p) for p in glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)})


ant_runs = {os.path.basename(d): d for d in find_dirs("outputs/*_seed[0-9]/repertoire.npz")}
ant_runs.update({os.path.basename(d): d for d in find_dirs("outputs/*_seed0_demo/repertoire.npz")})
ant_robust = {os.path.basename(d): d for d in find_dirs("outputs/rl_robust/seed[0-9]/policy.npz")}
print("ant repertoires:", sorted(ant_runs), "robust:", sorted(ant_robust), flush=True)

jobs = []
# Ant banks first: they are needed to validate the offline replay and to tune the methods.
names = sorted(ant_runs)
for half in (names[: len(names) // 2], names[len(names) // 2 :]):
    runs = " ".join(ant_runs[n] for n in half)
    jobs.append((f"bank_ant_{half[0]}", ["bash", "-c", f"{py} -m qd_damage.bank --runs {runs} --out {OUT}/bank"]))
robust = " ".join(ant_robust.values())
jobs.append(("bank_ant_robust", ["bash", "-c", f"{py} -m qd_damage.bank --robust {robust} --out {OUT}/bank"]))
for robot in ("walker2d", "halfcheetah"):
    steps = []
    for algo in ("dcrlme", "pgame", "me"):
        steps.append(f"{py} -m qd_damage.repertoires --algo {algo} --seed 0 --profile demo --robot {robot} "
                     f"--out {OUT}/repertoires")
        steps.append(f"{py} -m qd_damage.bank --runs {OUT}/repertoires/{robot}_{algo}_seed0_demo --out {OUT}/bank")
    jobs.append((f"dev_{robot}", ["bash", "-c", " && ".join(steps)]))
for robot in ("walker2d", "halfcheetah"):
    for seed in (0, 1):
        jobs.append((
            f"robust_{robot}_s{seed}",
            ["bash", "-c", f"{py} -m qd_damage.rl_retrain --robust --robot {robot} --seed {seed} --out {OUT}/rl_robust"
                           f" && {py} -m qd_damage.bank --robust {OUT}/rl_robust/{robot}_seed{seed} --out {OUT}/bank"],
        ))
ku.run_jobs(jobs, CODE, env, log_dir="/kaggle/working/logs")

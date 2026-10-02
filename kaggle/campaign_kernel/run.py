"""Kaggle (2 x T4): evaluation campaign on the 9 repertoires (outputs of the qd-damage-recovery-full kernel).

- for each repertoire: oracle (every policy x every damage), then adaptation (ITE, top-k, random, best-intact;
  5 repetitions; real simulated episodes);
- TD3 re-training on 4 damages x 2 variants (from scratch / from the best intact DCRL-ME policy) x 2 seeds.
  Fine-tuning runs after the oracle of its repertoire, so that it starts from the best RE-EVALUATED intact policy.
Two jobs at a time (one per GPU). Outputs in /kaggle/working/outputs/{oracle,adaptation,rl}/.
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
OUT = "/kaggle/working/outputs"
runs = sorted(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/outputs/*/repertoire.npz", recursive=True))
print("repertoires:", [os.path.basename(r) for r in runs], flush=True)
by_name = {os.path.basename(r): r for r in runs}
py = sys.executable
RL_SCENARIOS = ["leg4_paralysed", "leg2_weak", "legs14_paralysed", "ankle1_paralysed"]


def rl_cmd(scenario, variant, seed):
    return (f"{py} -m qd_damage.rl_retrain --scenario {scenario} --variant {variant} --seed {seed} "
            f"--repertoire {by_name[f'dcrlme_seed{seed}']} --oracle-root {OUT}/oracle --out {OUT}/rl")


jobs = []
for name, run_dir in by_name.items():
    steps = [f"{py} -m qd_damage.oracle --runs {run_dir} --out {OUT}/oracle",
             f"{py} -m qd_damage.adaptation --runs {run_dir} --oracle-root {OUT}/oracle --out {OUT}/adaptation"]
    if name in ("dcrlme_seed0", "dcrlme_seed1"):
        steps += [rl_cmd(sc, "finetune", int(name[-1])) for sc in RL_SCENARIOS]
    jobs.append((f"post_{name}", ["bash", "-c", " && ".join(steps)]))
for scenario in RL_SCENARIOS:
    for seed in [0, 1]:
        jobs.append((f"rl_{scenario}_scratch_s{seed}", ["bash", "-c", rl_cmd(scenario, "scratch", seed)]))
ku.run_jobs(jobs, CODE, env, log_dir="/kaggle/working/logs")

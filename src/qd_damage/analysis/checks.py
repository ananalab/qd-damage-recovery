"""Robustness checks of the main results.

- unbiased oracle: the best cell of each scenario re-evaluated on fresh episodes (oracle.check_best_cells);
- mixed models: algorithm effects with a random intercept per repertoire, as a check of the rank tests;
- reality gap: ITE and baselines when the trials run on a robot whose physics differ from the simulator;
- descriptor ablation: ITE with a Gaussian process on joint usage instead of feet contact;
- stronger RL baselines: fine-tuning with a critic warm-up or a pre-trained critic, and a damage-robust policy.

Each part is skipped when its results are missing. Every function returns (tables, macros); the LaTeX tables are
written in `tex_dir` when given.
"""

import glob
import re
from pathlib import Path


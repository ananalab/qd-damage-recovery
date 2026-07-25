"""Kaggle (2 x T4): the 9 full repertoires, 3 algorithms x 3 seeds, 2000 iterations each.

Two runs at a time (one per GPU), longest first (DCRL-ME, then PGA-ME, then MAP-Elites).
Outputs in /kaggle/working/outputs/<algo>_seed<seed>/, logs in /kaggle/working/logs/.
"""

import glob
import os
import shutil
import sys
import tarfile


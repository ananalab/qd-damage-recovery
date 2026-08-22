"""Statistics on (repertoire, damage) units: medians, Mann-Whitney and Wilcoxon tests, mixed models."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

from qd_damage.analysis.data import ALGOS, UNIT



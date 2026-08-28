"""Campaign analysis: tables, statistics, figures and LaTeX numbers, computed from results/ only.

Usage:
    python -m qd_damage.analysis.report          # final campaign: results/analysis, media/final, report/generated
    python -m qd_damage.analysis.report --dev    # development runs (*_demo): results/analysis_dev, media/dev

Reads results/repertoires/<run>/{run.json,metrics.csv}, results/oracle/<run>/summary.csv,
results/adaptation/<run>/{summary.csv,trials.csv}, results/rl/*/{curve.csv,rl.json} and results/bonus/.

Statistics. The five repetitions of an adaptation only differ by episode noise, so they are averaged first:
the statistical unit is a (repertoire, damage) pair, i.e. 3 seeds x 12 damages = 36 units per algorithm.
Algorithms are compared with Mann-Whitney tests on these units; ITE and top-k (or grid and continuous ITE)
are compared with paired Wilcoxon signed-rank tests. Units of the same seed share a repertoire, so they are
not fully independent and p-values are indicative.
"""

import argparse

import pandas as pd


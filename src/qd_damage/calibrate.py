"""Choice of the ITE Gaussian-process hyper-parameters by maximum marginal likelihood.

Usage:
    python -m qd_damage.calibrate --oracles results/oracle/me_seed0_demo results/oracle/pgame_seed0_demo ...
    python -m qd_damage.calibrate --joint-usage --oracles ...     # same, on the joint-usage descriptor

Data: development repertoires only (the 200-iteration demo runs), never those of the final campaign. For each
(repertoire, damage) pair, the oracle gives normalised residuals r = (damaged - intact) / scale on a random subset
of cells; the log marginal likelihood is summed over all pairs and the best (length-scale, signal variance,
noise variance) on a grid is kept. The chosen values are then frozen in configs/adaptation.yaml.

The script also reports the likelihood of a model without any correlation between cells (independent noise). The
gap between the two measures how much one trial tells about the other cells, i.e. how much ITE can generalise.
"""


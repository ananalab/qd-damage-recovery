"""Intelligent Trial & Error: M-BOA (Map-Based Bayesian Optimisation Algorithm), Cully et al., Nature 2015.

The repertoire gives, for each cell, the performance measured on the intact robot (the prior). After damage,
we try a gait, observe its performance, and a Gaussian process corrects the predictions of all similar gaits
(close in descriptor space). The next trial maximises an upper confidence bound, and the search stops as soon
as a tested gait reaches a fraction alpha of the best performance still expected.

Plain numpy: there are at most a few dozen trials. Performances are normalised (see `Normalizer`) so that the
parameters of Cully et al. (kappa, noise, length-scale) keep their meaning.
"""


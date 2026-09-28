"""Target transforms for CH4 selectivity.

Most CH4 selectivities sit near the 100 % ceiling, which makes a direct
regression poorly conditioned. Three parameterisations are compared:

- "direct" : predict the percentage, clipped to [0, 100]
- "logit"  : predict logit(S/100), invert with the sigmoid
- "impurity": predict log1p(100 - S), i.e. the non-CH4 share
"""

import numpy as np

EPS = 0.005


def _logit(y):
    p = np.clip(np.asarray(y, dtype=float) / 100.0, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _inv_logit(z):
    return 100.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def _impurity(y):
    return np.log1p(np.clip(100.0 - np.asarray(y, dtype=float), 0, 100))


def _inv_impurity(z):
    return np.clip(100.0 - np.expm1(np.asarray(z, dtype=float)), 0, 100)


def _identity(y):
    return np.asarray(y, dtype=float)


def _clip(z):
    return np.clip(np.asarray(z, dtype=float), 0, 100)


CH4_TRANSFORMS = {
    "direct": (_identity, _clip),
    "logit": (_logit, _inv_logit),
    "impurity": (_impurity, _inv_impurity),
}

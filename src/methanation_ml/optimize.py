"""Search operating/preparation conditions that maximise predicted performance.

Uses SciPy's differential evolution on top of the trained models. The model
can only interpolate: bounds should stay inside the ranges covered by the
training data, and any suggestion is a candidate for experimental testing,
not a validated optimum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from .predict import predict

WEIGHTS = {
    "balanced": (0.5, 0.5),
    "co2_priority": (0.7, 0.3),
    "ch4_priority": (0.3, 0.7),
    "co2_only": (1.0, 0.0),
    "ch4_only": (0.0, 1.0),
}


def optimise(bundle: dict, fixed: dict, bounds: dict[str, tuple[float, float]],
             strategy: str = "balanced", seed: int = 42, maxiter: int = 200,
             popsize: int = 25) -> pd.Series:
    w_co2, w_ch4 = WEIGHTS[strategy]
    names = list(bounds)

    def row(x):
        return pd.DataFrame([{**fixed, **dict(zip(names, x))}])

    def objective(x):
        p = predict(bundle, row(x)).iloc[0]
        return -(w_co2 * p["pred_CO2_conversion_pct"] + w_ch4 * p["pred_CH4_selectivity_pct"])

    res = differential_evolution(objective, bounds=[bounds[n] for n in names],
                                 strategy="best1bin", maxiter=maxiter, popsize=popsize,
                                 tol=0.01, mutation=(0.5, 1), recombination=0.7,
                                 seed=seed, polish=True)
    best = row(res.x)
    pred = predict(bundle, best).iloc[0]
    return pd.concat([best.iloc[0][names], pred,
                      pd.Series({"score": -res.fun, "strategy": strategy})])

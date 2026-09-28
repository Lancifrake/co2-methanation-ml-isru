"""Interpretability: SHAP values, rank correlations, oxide interactions.

These describe associations learned from a small literature dataset; they
are hypotheses to test experimentally, not causal effects.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from catboost import Pool
from scipy.stats import spearmanr

from .schema import OXIDE_COLS


def shap_importance(model, X: pd.DataFrame, cat_idx: list[int]) -> pd.DataFrame:
    """Mean |SHAP| per feature, using CatBoost's native TreeSHAP."""
    shap = model.get_feature_importance(Pool(X, cat_features=cat_idx), type="ShapValues")
    values = np.abs(shap[:, :-1]).mean(axis=0)  # last column is the expected value
    return (pd.DataFrame({"feature": X.columns, "mean_abs_shap": values})
            .sort_values("mean_abs_shap", ascending=False, ignore_index=True))


def spearman_vs_targets(X: pd.DataFrame, targets: dict[str, pd.Series],
                        min_obs: int = 10) -> pd.DataFrame:
    """Spearman rho and p-value of each numeric feature against each target."""
    rows = []
    for col in X.select_dtypes(include=[np.number]).columns:
        row = {"feature": col}
        for name, y in targets.items():
            mask = X[col].notna() & y.notna()
            if mask.sum() < min_obs or X.loc[mask, col].nunique() < 2:
                row[f"rho_{name}"], row[f"p_{name}"] = np.nan, np.nan
                continue
            rho, p = spearmanr(X.loc[mask, col], y[mask])
            row[f"rho_{name}"], row[f"p_{name}"] = rho, p
        rows.append(row)
    return pd.DataFrame(rows)


def oxide_pair_products(X: pd.DataFrame, targets: dict[str, pd.Series],
                        min_obs: int = 10) -> pd.DataFrame:
    """Spearman correlation of pairwise oxide products (wt% x wt%) with targets."""
    oxides = [c for c in OXIDE_COLS if c in X.columns]
    products = pd.DataFrame({f"{a} x {b}": X[a] * X[b] for a, b in combinations(oxides, 2)})
    out = spearman_vs_targets(products, targets, min_obs=min_obs)
    return out.rename(columns={"feature": "pair"})

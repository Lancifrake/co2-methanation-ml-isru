"""Inference with a saved model bundle."""

from __future__ import annotations

import pandas as pd

from .features import build_features
from .targets import CH4_TRANSFORMS


def predict(bundle: dict, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Predict CO2 conversion and CH4 selectivity (%) for new conditions.

    Missing input columns are allowed (CatBoost treats them as missing).
    Derived features are recomputed automatically.
    """
    X = build_features(new_rows, columns=bundle["columns"])
    _, inverse = CH4_TRANSFORMS[bundle["ch4_variant"]]
    return pd.DataFrame({
        "pred_CO2_conversion_pct": bundle["co2_model"].predict(X).clip(0, 100),
        "pred_CH4_selectivity_pct": inverse(bundle["ch4_model"].predict(X)),
    }, index=new_rows.index)

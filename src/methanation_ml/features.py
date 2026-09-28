"""Loading, cleaning and feature engineering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import (
    CATEGORICAL_COLS, DERIVED_COLS, GROUP_COL, ID_COLS, NUMERIC_COLS, TARGET_CH4,
    TARGET_CO2,
)

_TARGET_ALIASES = {
    TARGET_CO2: ["CO2 Conversion %", "CO₂ Conversion %", "CO2 conversion %", "CO2 Conversion%"],
    TARGET_CH4: ["CH4 selectivity %", "CH₄ selectivity %", "CH4 Selectivity %", "CH4 selectivity%"],
}


def load_table(path: str | Path) -> pd.DataFrame:
    """Read the dataset from .xlsx (sheet 'Data' if present) or .csv."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None)
        return sheets.get("Data", next(iter(sheets.values())))
    return pd.read_csv(path)


def _normalise_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Map target-column spelling variants to the canonical names."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    renames = {}
    for canonical, aliases in _TARGET_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lookup:
                renames[lookup[alias.lower()]] = canonical
                break
    return df.rename(columns=renames)


def add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """Physically motivated features used in all models.

    - Temperature_sq : captures the bell-shaped conversion vs. temperature curve
    - inv_WF         : inverse weight/flow ratio (proxy for space velocity)
    - pCO2_bar/pH2_bar : partial pressures from total pressure and CO2/H2 ratio
    """
    X = X.copy()
    num = lambda c: pd.to_numeric(X[c], errors="coerce") if c in X else np.nan

    X["Temperature_sq"] = num("Temperature (C)") ** 2
    wf = num("W/F (mgcat/minml)")
    X["inv_WF"] = 1.0 / wf.replace(0, np.nan) if isinstance(wf, pd.Series) else np.nan

    ratio, pressure = num("Ration CO2/H2"), num("Pressure (bar)")
    if isinstance(ratio, pd.Series) and isinstance(pressure, pd.Series):
        den = (1.0 + ratio).replace(0, np.nan)
        X["pCO2_bar"] = pressure * ratio / den
        X["pH2_bar"] = pressure / den
    else:
        X["pCO2_bar"] = np.nan
        X["pH2_bar"] = np.nan
    return X


def coerce_types(X: pd.DataFrame) -> pd.DataFrame:
    """Numeric columns -> float (text becomes NaN); everything else -> str."""
    X = X.copy()
    for c in X.columns:
        if c in NUMERIC_COLS:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        else:
            X[c] = X[c].astype(str)
    return X


def build_features(X: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Type coercion + derived features, optionally aligned to a column list."""
    X = add_derived_features(coerce_types(X))
    if columns is not None:
        X = X.reindex(columns=columns)
        for c in columns:
            if c not in NUMERIC_COLS and c not in DERIVED_COLS:
                X[c] = X[c].astype(str)
    return X


def prepare_dataset(df: pd.DataFrame):
    """Return X, y_co2, y_ch4, groups from the raw table.

    Rows missing either target are dropped. The literature reference is used
    only to define LOGO groups and is never a model input.
    """
    df = _normalise_headers(df)
    missing = [c for c in (TARGET_CO2, TARGET_CH4, GROUP_COL) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[TARGET_CO2, TARGET_CH4]).reset_index(drop=True)
    drop = {TARGET_CO2, TARGET_CH4, GROUP_COL, *ID_COLS}
    X = build_features(df.drop(columns=[c for c in df.columns if c in drop]))
    return (
        X,
        df[TARGET_CO2].astype(float),
        df[TARGET_CH4].astype(float),
        df[GROUP_COL].astype(str),
    )


def categorical_indices(X: pd.DataFrame) -> list[int]:
    return [i for i, c in enumerate(X.columns) if X[c].dtype == object]


__all__ = [
    "load_table", "prepare_dataset", "build_features", "add_derived_features",
    "coerce_types", "categorical_indices", "CATEGORICAL_COLS",
]

"""Leave-One-Group-Out validation of CatBoost models.

Each literature source is one group: models are trained on all sources but
one and evaluated on the held-out source, which measures transfer to an
unseen experimental campaign.

Early stopping and the choice of CH4 target transform are decided with an
*inner* LOGO loop on the training sources only, so the held-out source is
never used for any modelling decision. (The exploratory notebooks used the
held-out source for early stopping; ``legacy_test_fold_early_stopping=True``
reproduces that behaviour for comparison only.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

from .features import categorical_indices
from .targets import CH4_TRANSFORMS

CO2_PARAMS = dict(depth=4, learning_rate=0.03, l2_leaf_reg=10, loss_function="RMSE")
CH4_PARAMS = dict(depth=6, learning_rate=0.06, l2_leaf_reg=10, subsample=0.8,
                  loss_function="RMSE")
MAX_ITER = 1200


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    r2 = r2_score(y_true, y_pred) if np.var(y_true) > 1e-8 else np.nan
    return {
        "R2": r2,
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def _fit(params, X, y, cat_idx, iterations, eval_set=None, seed=42, patience=150):
    model = CatBoostRegressor(**params, iterations=iterations, random_seed=seed,
                              verbose=False, allow_writing_files=False)
    kw = {}
    if eval_set is not None:
        kw = dict(eval_set=eval_set, use_best_model=True, early_stopping_rounds=patience)
    model.fit(X, y, cat_features=cat_idx, **kw)
    return model


def _inner_search(params, X, y_fit, y_true_pct, inverse, groups, cat_idx, seed):
    """Inner LOGO on training sources: best iteration count + error in % units."""
    iters, errors = [], []
    for tr, va in LeaveOneGroupOut().split(X, y_fit, groups):
        m = _fit(params, X.iloc[tr], y_fit[tr], cat_idx, MAX_ITER,
                 eval_set=(X.iloc[va], y_fit[va]), seed=seed)
        iters.append(max(m.get_best_iteration() or 1, 1) + 1)
        errors.append(metrics(y_true_pct[va], inverse(m.predict(X.iloc[va])))["RMSE"])
    return int(np.median(iters)), float(np.mean(errors))


@dataclass
class LogoResult:
    per_fold: pd.DataFrame
    predictions: pd.DataFrame
    ch4_variant_votes: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Mean ± std across folds, and pooled out-of-fold metrics."""
        rows = []
        p = self.predictions
        for target, t, pr in [("CO2", "y_CO2_true", "y_CO2_pred"),
                              ("CH4", "y_CH4_true", "y_CH4_pred")]:
            pooled = metrics(p[t], p[pr])
            for m in ("R2", "RMSE", "MAE"):
                col = self.per_fold[f"{target}_{m}"]
                rows.append({"target": target, "metric": m,
                             "fold_mean": col.mean(), "fold_std": col.std(ddof=0),
                             "pooled_oof": pooled[m]})
        return pd.DataFrame(rows)


def run_logo(X: pd.DataFrame, y_co2: pd.Series, y_ch4: pd.Series, groups: pd.Series,
             seed: int = 42, legacy_test_fold_early_stopping: bool = False) -> LogoResult:
    cat_idx = categorical_indices(X)
    y_co2, y_ch4 = y_co2.to_numpy(float), y_ch4.to_numpy(float)
    fold_rows, pred_frames, votes = [], [], {}

    for fold, (tr, te) in enumerate(LeaveOneGroupOut().split(X, y_co2, groups), start=1):
        Xtr, Xte, gtr = X.iloc[tr], X.iloc[te], groups.iloc[tr].reset_index(drop=True)
        Xtr = Xtr.reset_index(drop=True)
        held_out = groups.iloc[te].iloc[0]

        # --- CO2 conversion ---
        if legacy_test_fold_early_stopping:
            m_co2 = _fit(CO2_PARAMS, Xtr, y_co2[tr], cat_idx, MAX_ITER,
                         eval_set=(Xte, y_co2[te]), seed=seed)
        else:
            n_iter, _ = _inner_search(CO2_PARAMS, Xtr, y_co2[tr], y_co2[tr],
                                      lambda z: z, gtr, cat_idx, seed)
            m_co2 = _fit(CO2_PARAMS, Xtr, y_co2[tr], cat_idx, n_iter, seed=seed)
        pred_co2 = m_co2.predict(Xte)

        # --- CH4 selectivity: pick the transform ---
        if legacy_test_fold_early_stopping:
            # notebook behaviour: best variant chosen on the held-out source
            best = None
            for name, (fwd, inv) in CH4_TRANSFORMS.items():
                m = _fit(CH4_PARAMS, Xtr, fwd(y_ch4[tr]), cat_idx, MAX_ITER,
                         eval_set=(Xte, fwd(y_ch4[te])), seed=seed, patience=200)
                p = inv(m.predict(Xte))
                score = metrics(y_ch4[te], p)["R2"]
                if best is None or np.nan_to_num(score, nan=-1e9) > best[1]:
                    best = (name, np.nan_to_num(score, nan=-1e9), p)
            variant, _, pred_ch4 = best
        else:
            candidates = {}
            for name, (fwd, inv) in CH4_TRANSFORMS.items():
                candidates[name] = _inner_search(CH4_PARAMS, Xtr, fwd(y_ch4[tr]),
                                                 y_ch4[tr], inv, gtr, cat_idx, seed)
            variant = min(candidates, key=lambda k: candidates[k][1])
            fwd, inv = CH4_TRANSFORMS[variant]
            m_ch4 = _fit(CH4_PARAMS, Xtr, fwd(y_ch4[tr]), cat_idx,
                         candidates[variant][0], seed=seed)
            pred_ch4 = inv(m_ch4.predict(Xte))
        votes[variant] = votes.get(variant, 0) + 1

        mc, mh = metrics(y_co2[te], pred_co2), metrics(y_ch4[te], pred_ch4)
        fold_rows.append({
            "fold": fold, "held_out_source": held_out, "n_test": len(te),
            **{f"CO2_{k}": v for k, v in mc.items()},
            "CH4_variant": variant,
            **{f"CH4_{k}": v for k, v in mh.items()},
            "CH4_min": y_ch4[te].min(), "CH4_max": y_ch4[te].max(),
        })
        pred_frames.append(pd.DataFrame({
            "fold": fold, "held_out_source": held_out,
            "y_CO2_true": y_co2[te], "y_CO2_pred": pred_co2,
            "y_CH4_true": y_ch4[te], "y_CH4_pred": pred_ch4,
        }))

    return LogoResult(pd.DataFrame(fold_rows), pd.concat(pred_frames, ignore_index=True),
                      votes)


def fit_final_models(X, y_co2, y_ch4, groups, ch4_variant: str, seed: int = 42) -> dict:
    """Refit on all sources; iteration counts come from a LOGO search."""
    cat_idx = categorical_indices(X)
    y_co2, y_ch4 = y_co2.to_numpy(float), y_ch4.to_numpy(float)
    g = groups.reset_index(drop=True)
    X = X.reset_index(drop=True)

    n_co2, _ = _inner_search(CO2_PARAMS, X, y_co2, y_co2, lambda z: z, g, cat_idx, seed)
    fwd, inv = CH4_TRANSFORMS[ch4_variant]
    n_ch4, _ = _inner_search(CH4_PARAMS, X, fwd(y_ch4), y_ch4, inv, g, cat_idx, seed)

    return {
        "co2_model": _fit(CO2_PARAMS, X, y_co2, cat_idx, n_co2, seed=seed),
        "ch4_model": _fit(CH4_PARAMS, X, fwd(y_ch4), cat_idx, n_ch4, seed=seed),
        "ch4_variant": ch4_variant,
        "columns": list(X.columns),
        "cat_idx": cat_idx,
        "iterations": {"co2": n_co2, "ch4": n_ch4},
    }

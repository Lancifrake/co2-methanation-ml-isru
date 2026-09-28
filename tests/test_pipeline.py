import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_synthetic_data import make  # noqa: E402

from methanation_ml.features import add_derived_features, prepare_dataset
from methanation_ml.predict import predict
from methanation_ml.targets import CH4_TRANSFORMS
from methanation_ml.validation import fit_final_models, run_logo


def test_ch4_transforms_round_trip():
    y = np.array([60.0, 90.0, 99.0])
    for fwd, inv in CH4_TRANSFORMS.values():
        assert np.allclose(inv(fwd(y)), y, atol=1e-6)


def test_derived_features():
    X = add_derived_features(pd.DataFrame({
        "Temperature (C)": [300.0], "W/F (mgcat/minml)": [4.0],
        "Pressure (bar)": [1.0], "Ration CO2/H2": [0.25],
    }))
    assert X["Temperature_sq"].iloc[0] == 90000
    assert X["inv_WF"].iloc[0] == 0.25
    assert np.isclose(X["pCO2_bar"].iloc[0] + X["pH2_bar"].iloc[0], 1.0)


def test_reference_is_not_a_feature():
    X, *_ = prepare_dataset(make(n_per_source=5))
    assert "Reference" not in X.columns


def test_logo_end_to_end():
    X, y_co2, y_ch4, groups = prepare_dataset(make(n_per_source=12))
    res = run_logo(X, y_co2, y_ch4, groups)
    assert len(res.per_fold) == groups.nunique()
    assert len(res.predictions) == len(X)
    bundle = fit_final_models(X, y_co2, y_ch4, groups, "logit")
    pred = predict(bundle, make(n_per_source=1).drop(columns=["CO2 Conversion %"]))
    assert pred["pred_CH4_selectivity_pct"].between(0, 100).all()

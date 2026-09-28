"""Predict CO2 conversion / CH4 selectivity for new conditions.

Example:
    python scripts/predict.py --model outputs/demo/model_bundle.joblib \
        --input my_conditions.xlsx --out predictions.csv
"""

import argparse

import joblib
import pandas as pd

from methanation_ml.features import load_table
from methanation_ml.predict import predict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True, help="rows using the dataset column names")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--write-template", action="store_true",
                    help="write an empty input template with the expected columns and exit")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    if args.write_template:
        raw_cols = [c for c in bundle["columns"]
                    if c not in {"Temperature_sq", "inv_WF", "pCO2_bar", "pH2_bar"}]
        pd.DataFrame(columns=raw_cols).to_csv(args.input, index=False)
        print(f"Template written to {args.input}")
        return

    rows = load_table(args.input).dropna(how="all")
    out = pd.concat([rows, predict(bundle, rows)], axis=1)
    out.to_csv(args.out, index=False)
    print(out.filter(like="pred_").round(2).to_string())


if __name__ == "__main__":
    main()

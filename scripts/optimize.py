"""Suggest conditions maximising predicted performance (differential evolution).

Example:
    python scripts/optimize.py --model outputs/demo/model_bundle.joblib \
        --config configs/optimize_example.json
"""

import argparse
import json

import joblib

from methanation_ml.optimize import optimise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = json.loads(open(args.config, encoding="utf-8").read())
    result = optimise(joblib.load(args.model), cfg["fixed"],
                      {k: tuple(v) for k, v in cfg["bounds"].items()},
                      strategy=cfg.get("strategy", "balanced"),
                      maxiter=cfg.get("maxiter", 200), popsize=cfg.get("popsize", 25))
    print(result.to_string())


if __name__ == "__main__":
    main()

"""LOGO-CV evaluation, final model fit and interpretability exports.

Example:
    python scripts/train.py --data data/sample/synthetic_example.csv --out outputs/demo
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from methanation_ml.features import load_table, prepare_dataset  # noqa: E402
from methanation_ml.interpret import (  # noqa: E402
    oxide_pair_products, shap_importance, spearman_vs_targets,
)
from methanation_ml.validation import fit_final_models, run_logo  # noqa: E402


def parity_plot(pred, target, path):
    t, p = pred[f"y_{target}_true"], pred[f"y_{target}_pred"]
    fig, ax = plt.subplots(figsize=(5, 5))
    for src, d in pred.groupby("held_out_source"):
        ax.scatter(d[f"y_{target}_true"], d[f"y_{target}_pred"], s=18, label=str(src))
    lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(f"Experimental {target} (%)")
    ax.set_ylabel(f"Predicted {target} (%) — held-out source")
    ax.legend(title="Held-out source", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def shap_bar(shap_df, target, path, top=12):
    d = shap_df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(d["feature"], d["mean_abs_shap"])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"{target}: top {top} features")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help=".xlsx or .csv with the dataset schema")
    ap.add_argument("--out", default="outputs/run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--legacy-test-fold-early-stopping", action="store_true",
                    help="Reproduce the exploratory notebooks (optimistic; comparison only)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    X, y_co2, y_ch4, groups = prepare_dataset(load_table(args.data))
    print(f"{len(X)} experiments, {groups.nunique()} sources, {X.shape[1]} features")

    res = run_logo(X, y_co2, y_ch4, groups, seed=args.seed,
                   legacy_test_fold_early_stopping=args.legacy_test_fold_early_stopping)
    summary = res.summary()
    res.per_fold.to_csv(out / "logo_per_fold.csv", index=False)
    res.predictions.to_csv(out / "logo_oof_predictions.csv", index=False)
    summary.to_csv(out / "logo_summary.csv", index=False)
    print(res.per_fold.round(3).to_string(index=False))
    print(summary.round(3).to_string(index=False))

    for target in ("CO2", "CH4"):
        parity_plot(res.predictions, target, out / f"parity_{target}_logo.png")

    variant = max(res.ch4_variant_votes, key=res.ch4_variant_votes.get)
    bundle = fit_final_models(X, y_co2, y_ch4, groups, variant, seed=args.seed)
    joblib.dump(bundle, out / "model_bundle.joblib")

    for target in ("CO2", "CH4"):
        shap_df = shap_importance(bundle[f"{target.lower()}_model"], X, bundle["cat_idx"])
        shap_df.to_csv(out / f"shap_{target}.csv", index=False)
        shap_bar(shap_df, target, out / f"shap_{target}.png")
    targets = {"CO2": y_co2, "CH4": y_ch4}
    spearman_vs_targets(X, targets).to_csv(out / "spearman_features.csv", index=False)
    oxide_pair_products(X, targets).to_csv(out / "spearman_oxide_pairs.csv", index=False)

    (out / "run_info.json").write_text(json.dumps({
        "n_experiments": len(X), "n_sources": int(groups.nunique()),
        "ch4_variant": variant, "ch4_variant_votes": res.ch4_variant_votes,
        "final_iterations": bundle["iterations"],
        "legacy_test_fold_early_stopping": args.legacy_test_fold_early_stopping,
    }, indent=2))
    print(f"Outputs written to {out}/")


if __name__ == "__main__":
    main()

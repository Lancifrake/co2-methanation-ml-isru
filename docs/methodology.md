# Methodology and limitations

## Why Leave-One-Group-Out

Rows from the same article share a catalyst family, a reactor and a protocol.
A random split puts near-duplicates in train and test and inflates scores
(a group hold-out Random Forest reached train R² ≈ 0.98 while failing on CH₄
in test). Holding out one whole source per fold asks the realistic question:
*how well does the model predict a campaign it has never seen?*

## Model choice

Several families were tried during the project: Random Forest (scikit-learn
pipeline with one-hot encoding and GridSearchCV over GroupKFold), LightGBM
(GroupKFold per source) and CatBoost. On LOGO/GroupKFold by source, LightGBM
gave CO₂ R² ≈ 0.69 but CH₄ R² ≈ −5.0 ± 5.0. CatBoost was kept because it
handles categorical columns and missing values natively and was the most
stable on CH₄, while remaining poor on that target.

Hyper-parameters (fixed): CO₂ depth 4, lr 0.03, l2 10; CH₄ depth 6, lr 0.06,
l2 10, subsample 0.8. Up to 1200 trees with early stopping.

## Nested decisions (no leakage)

Inside each outer fold, an inner LOGO over the three training sources gives:

- the number of trees (median best iteration across inner folds);
- the CH₄ parameterisation (direct / logit / log-impurity) with the lowest
  inner RMSE in % units.

The model is then refit on the three sources and evaluated once on the
held-out source.

## Metrics

Both are reported because they tell different stories:

- **mean ± std of per-fold R²** — how stable the model is across sources;
- **pooled out-of-fold R²/RMSE/MAE** — all 104 held-out predictions together.

For CH₄, per-fold R² is very negative because within-source variance is small
(values near 100 %); MAE is the more interpretable figure.

## Interpretability

- CatBoost TreeSHAP (mean |SHAP|) on the final models.
- Spearman correlations of numeric features and of pairwise oxide products
  with each target.

Oxide composition is almost constant within a source, so oxide effects are
confounded with the source itself. Treat these as leads for experiments.

## Condition search

`optimize.py` runs differential evolution on the model predictions. Bounds
must stay inside the training ranges; categorical values not seen in training
are ignored by the model. Outputs are candidates to test, not validated optima.

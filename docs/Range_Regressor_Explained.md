# Range Regressor — Plain English Explanation

> **File this explains:** `range_regressor.py`  
> **Prerequisite knowledge:** None — this starts from scratch.

---

## What Does "Range" Mean Here?

**Range = the physical distance (metres) between the UWB transmitter and receiver.**

The dataset contains a `RANGE` column — that is the measured ground truth for Path 1. Path 2's range is not directly in the CSVs and is estimated from the CIR by extracting the second peak and converting the sample gap into metres.

---

## Why Two Regressors?

The project requires range estimates for the two dominant paths (first and second arrivals):

- Path 1: the first arrival — its measured range is the `RANGE` column.
- Path 2: the second arrival — not provided directly; we detect a second peak and compute an estimated target `RANGE_PATH2_EST = RANGE + PEAK2_GAP × 0.15m`.

Therefore we train two separate regressors using the same feature matrix but different targets:

- `rf_range_path1.pkl` → predicts Path 1 (measured) range
- `rf_range_path2.pkl` → predicts Path 2 (estimated) range

Both models are Random Forest regressors in `range_regressor.py`.

---

## Data Inputs (what the regressor uses)

- `X_train_regression.npy`, `X_test_regression.npy`: feature matrices (139 features)
  - 19 core features: hardware/signal measures like `FP_IDX`, `FP_AMP1/2/3`, `SNR`, `RXPACC`, `CIR_PWR`, and engineered second-path features (`PEAK2_IDX`, `PEAK2_AMP`, `PEAK2_GAP`, `PEAK2_FOUND`).
  - 120 CIR samples: amplitudes from CIR indices 730–849 (the active window).
- Targets:
  - `y_range_p1_train.npy`, `y_range_p1_test.npy` — Path 1 measured ranges
  - `y_range_p2_train.npy`, `y_range_p2_test.npy` — Path 2 estimated ranges

The training/test split and scalers are created upstream in `second_path_features.py` / `preprocess_data.py`.

---

## Model Choice: Random Forest Regressor

Why Random Forest?
- Handles nonlinear relationships between CIR features and range.
- Robust to outliers and unscaled features (no mandatory scaling required).
- Provides feature importance to interpret what drives predictions.
- Efficient to train and parallelizable with `n_jobs=-1`.

Typical training call used in `range_regressor.py`:

```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
```

Two independent models are trained (one per path target).

---

## Training & Evaluation Steps (what the script does)

1. Load X_train, X_test and the two y targets.
2. Initialize two RandomForestRegressor instances (same hyperparameters by default).
3. Fit model A on (X_train, y_p1_train); fit model B on (X_train, y_p2_train).
4. Predict on X_test for both models.
5. Compute regression metrics: RMSE, MAE, R².
6. Save the trained models to `models/range_regressor/rf_range_path1.pkl` and `rf_range_path2.pkl`.
7. Produce diagnostic plots: predicted vs actual, residual histograms, absolute error vs range, and feature importance plots.

---

## Interpreting the Targets & Errors

- Path 1 target (`RANGE`) is measured (ground truth). Errors measure how well the model reproduces the measured distance.
- Path 2 target (`RANGE_PATH2_EST`) is derived (estimated from `PEAK2_GAP`). Any error in peak detection propagates into the target and thus into evaluation.

Metrics reported in the repo (example):
- Path 1 RMSE ≈ 1.28 m, MAE ≈ 0.98 m, R² ≈ 0.71
- Path 2 RMSE ≈ 1.32 m, MAE ≈ 1.01 m, R² ≈ 0.74

Note: Path 2 can have slightly higher RMSE yet higher R² because its values may span a wider dynamic range (R² is relative to variance).

---

## Important Features to Inspect

Feature importance usually highlights:
- `FP_IDX`: direct time-of-arrival indicator — very predictive for Path 1.
- `PEAK2_GAP` / `PEAK2_IDX`: directly encode the second arrival timing — key for Path 2.
- `CIR_PWR`, `FP_AMP1/2/3`, `RXPACC`: correlate with signal strength/quality and indirectly with range.
- Certain CIR sample indices (raw amplitudes) can also be informative when combined by the trees.

Use the saved `feature_importance_ranking.csv` (in `models/random_forest/` or `models/range_regressor/`) to pick top features for analysis or a compact model.

---

## Practical tips to improve the regressor

1. Clean target noise: improve peak detection parameters (prominence/height/min_gap) in `second_path_features.py` to reduce target noise for Path 2.
2. Hyperparameter tuning: grid search on `n_estimators`, `max_depth`, `min_samples_leaf`.
3. Feature selection: train with top-N features to reduce complexity and overfitting.
4. Cross-validation: use K-fold CV to get robust error estimates (instead of a single 80/20 split).
5. Ensembles: combine RF with other regressors (e.g., gradient boosting) or average predictions.

---

## Where to Look in the Repo

- `second_path_features.py` — how `PEAK2_*` features and `RANGE_PATH2_EST` are created.
- `range_regressor.py` — training, evaluation, saving models and plots.
- `preprocessed_data/` — `.npy` inputs used by the regressor.
- `models/range_regressor/` — saved `.pkl` models and diagnostic plots.

---

## Quick checklist to re-run training locally

1. Ensure `preprocessed_data/` and `models/` exist and contain required `.npy` files.
2. (Optional) Adjust peak extraction parameters in `second_path_features.py` and re-run it to regenerate targets.
3. Run `python range_regressor.py` from the repo root.
4. Check `models/range_regressor/` for `rf_range_path1.pkl`, `rf_range_path2.pkl` and the output plots.

---

## Final notes

- Path 2 range is an estimate derived from timing; treat its ground truth as noisy. Improving peak extraction will directly improve Path 2 regression performance.
- Random Forest gives solid baseline performance and interpretable feature importance — good for the 3D process emphasis in your project.



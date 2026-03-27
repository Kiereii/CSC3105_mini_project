# XGBoost Range Regressor Pipeline (Pseudocode)

## 0) Purpose and Prerequisites
- Define pipeline as standalone XGBoost-based two-target regressor for UWB ranges:
  - `Path 1` target: `y_range_p1`
  - `Path 2` target: `y_range_p2`
- Require preprocessed `.npy` inputs generated earlier by `second_path_features.py`.
- Import runtime, math/data, plotting, model, CV, and serialization dependencies.
- If XGBoost import fails, stop with runtime error indicating dependency is required.
- Silence warnings globally.

---

## 1) Configuration and Environment Setup
- Read environment variables (with defaults):
  - `RUN_NAME` default `"split_env_70_15_15_seed42"`
  - `RANDOM_SEED` default `42`
  - `XGB_TREE_METHOD` default `"hist"`
  - `RUN_KFOLD` enabled only when env string equals `"1"`
  - `CV_SPLITS` default `5`
  - `CV_N_JOBS` default `-1`
- Build directories:
  - `DATA_DIR = ./runs/{RUN_NAME}/preprocessed_data`
  - `OUTPUT_DIR = ./runs/{RUN_NAME}/models/range_regressor`
- Ensure `OUTPUT_DIR` exists.
- Print startup banner and CV status.

---

## 2) Load Data (Path 1 and Path 2 targets)
- Load regression arrays from `DATA_DIR`:
  - `X_train_regression.npy -> X_train`
  - `X_test_regression.npy -> X_test`
  - `y_range_p1_train.npy -> y_p1_train`
  - `y_range_p1_test.npy -> y_p1_test`
  - `y_range_p2_train.npy -> y_p2_train`
  - `y_range_p2_test.npy -> y_p2_test`
- Print dataset summary:
  - training sample count
  - test sample count
  - feature count
  - observed min/max range for Path 1 and Path 2 (training set)

---

## 3) Helper Utilities and Model Configuration
- `evaluate_metrics(y_true, y_pred)`:
  - Compute `RMSE`, `MAE`, `R2`
  - Return all as float values
- `to_float(value, default=NaN)`:
  - Convert safely; fallback to default on parse failure
- `to_int(value, default=0)`:
  - Convert safely; fallback to default on parse failure
- `build_xgb_regressor()` returns configured `XGBRegressor`:
  - objective: squared error regression
  - eval metric: RMSE
  - hyperparameters:
    - `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`
    - `subsample=0.9`, `colsample_bytree=0.9`
    - `reg_alpha=0`, `reg_lambda=1`
    - `tree_method = XGB_TREE_METHOD`
    - `random_state = RANDOM_SEED`
    - `n_jobs=-1`, quiet verbosity
- `fit_and_evaluate(path_label, model, X_tr, y_tr, X_te, y_te)`:
  - Fit model; time training
  - Predict on test; time inference
  - Compute metrics with `evaluate_metrics`
  - Print timing and metrics
  - Return dictionary with model, predictions, timings, metrics
- `run_kfold_cv(path_label, X_tr, y_tr)`:
  - Configure `KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)`
  - Use scoring:
    - RMSE as negative root MSE
    - MAE as negative MAE
    - R2
  - Run `cross_validate` on a fresh `build_xgb_regressor()`
  - Convert negative error scores to positive magnitudes
  - Compute mean/std for RMSE, MAE, R2
  - Print CV summary
  - Return summary dictionary

---

## 4) Train and Evaluate for Both Paths
- Initialize `cv_p1`, `cv_p2` as null.
- If `RUN_KFOLD` is enabled:
  - Run CV for Path 1 on `(X_train, y_p1_train)`
  - Run CV for Path 2 on `(X_train, y_p2_train)`
- Train/evaluate main models:
  - `p1_result = fit_and_evaluate(... y_p1_train/y_p1_test ...)`
  - `p2_result = fit_and_evaluate(... y_p2_train/y_p2_test ...)`
- Store in `results = {"p1": p1_result, "p2": p2_result}`.
- Build `results_df` with one row per path including:
  - model identifiers (`XGBoost`, `xgb`)
  - path label
  - RMSE/MAE/R2
  - train and predict durations
  - CV metadata (`cv_used`, `cv_splits`, and CV means/stds when available)
- Save comparison table:
  - `xgb_regression_model_comparison.csv`

---

## 5) Enforce Physical Constraint (`pred_p2 >= pred_p1`)
- Read predictions:
  - `p1_pred = results["p1"]["y_pred"]`
  - `p2_pred = results["p2"]["y_pred"]`
- Count violations where `p2_pred < p1_pred`.
- Apply clipping:
  - `p2_clipped = max(p2_pred, p1_pred)` elementwise
  - Replace `results["p2"]["y_pred"]` with clipped values
- If any violations existed:
  - Recompute Path 2 RMSE/MAE/R2 against `y_p2_test`
  - Update Path 2 metrics in both `results` and `results_df`
  - Print clipped count and new Path 2 RMSE
- Else print that no violations were found.
- Re-save `xgb_regression_model_comparison.csv` with updated metrics.

---

## 6) Visualization Outputs (Step 3)

### 6A) Predicted vs Actual Scatter (Both Paths)
- Create 2-panel scatter figure:
  - Left: Path 1 actual (`y_p1_test`) vs predicted (`results["p1"]["y_pred"]`)
  - Right: Path 2 actual (`y_p2_test`) vs predicted (`results["p2"]["y_pred"]`)
- Overlay diagonal identity line (`ideal prediction`).
- Include RMSE/MAE/R2 in each panel title.
- Save:
  - `xgb_range_estimation_results.png`
- Represents:
  - calibration/fit quality and bias around perfect-fit line.

### 6B) Metrics Comparison by Path
- Extract Path 1/2 values from `results_df` for RMSE, MAE, R2.
- Create 2-panel line plot:
  - Left panel: RMSE and MAE versus path (same unit: meters)
  - Right panel: R2 versus path (separate y-scale up to 1.0)
- Annotate numeric values at points.
- Save:
  - `xgb_regression_metrics_comparison.png`
- Represents:
  - side-by-side performance contrast between path targets.

### 6C) Feature Importance (Top 20 per Path Model)
- Build feature names:
  - predefined `core_feats` list
  - plus CIR features `CIR730` to `CIR849`
  - fallback to generic names `f0..fN-1` if feature count mismatch
- Determine key column indices when present:
  - `FP_IDX_col`, `PEAK2_GAP_col`
- Create 2-panel horizontal bar charts:
  - Path 1 model top-20 importances
  - Path 2 model top-20 importances
- Save:
  - `xgb_regressor_feature_importance.png`
- Represents:
  - most influential input features per target model.

### 6D) FP_IDX vs Measured Range Diagnostic (conditional)
- Load pair labels for test split:
  - prefer `y_test_pair.npy`
  - fallback `y_test.npy`
  - semantics: `0 = LOS+NLOS`, `1 = NLOS+NLOS`
- If `FP_IDX` feature exists:
  - Plot FP index (`X_test[:, FP_IDX_col]`) vs measured range, separately for Path 1 and Path 2
  - Color points by pair class (`LOS+NLOS`, `NLOS+NLOS`)
  - Fit and draw global linear trend line per panel
- Save:
  - `xgb_fp_idx_vs_range.png`
- Represents:
  - correlation of first-path CIR index delay with measured range; includes class-wise spread.

### 6E) PEAK2_GAP vs Extra Path Length Diagnostic (conditional)
- If `PEAK2_GAP` exists:
  - Set `METERS_PER_SAMPLE = (3e8 * 1.0016e-9) / 2`
  - Compute:
    - `peak2_gap_test = X_test[:, PEAK2_GAP_col]`
    - `extra_range_test = y_p2_test - y_p1_test`
  - Scatter by pair class (`LOS+NLOS`, `NLOS+NLOS`)
  - Overlay theoretical line: `extra_range = PEAK2_GAP * METERS_PER_SAMPLE`
- Save:
  - `xgb_peak2_gap_vs_extra_range.png`
- Represents:
  - validation that engineered peak gap corresponds to physical extra path length.

### 6F) Residual Distribution Histograms
- Compute residuals:
  - `p1_residuals = pred_p1 - actual_p1`
  - `p2_residuals = pred_p2 - actual_p2` (after clipping for Path 2)
- Create 2-panel histograms:
  - one per path with zero-error vertical line
  - mean-error vertical line and legend
  - title includes RMSE and residual skewness
- Save:
  - `xgb_residual_distribution.png`
- Represents:
  - error spread, central tendency, asymmetry, and bias direction.

### 6G) Predicted Path 1 vs Predicted Path 2
- Scatter `pred_p1` (x-axis) vs `pred_p2` (y-axis), colored by pair class.
- Draw diagonal constraint boundary line `P2 = P1`.
- Set equal axis limits from combined prediction range.
- Save:
  - `xgb_p1_vs_p2_predicted.png`
- Represents:
  - co-variation between path predictions and visual check of enforced physical rule (`P2 >= P1`).

---

## 7) Save Models, Predictions, and Text Report

### 7A) Persist Models and Prediction Arrays
- Save trained models:
  - `xgb_range_path1.pkl`
  - `xgb_range_path2.pkl`
- Save test predictions:
  - `y_p1_pred_xgb.npy`
  - `y_p2_pred_xgb.npy` (post-constraint Path 2 predictions)

### 7B) Write Final Report
- Create `xgb_regression_results.txt` containing:
  - header and dataset sizes
  - per-path performance line (RMSE/MAE/R2/train time)
  - optional CV summary per path (if enabled and values finite)
  - pair-class composition on test split:
    - LOS+NLOS count and percent
    - NLOS+NLOS count and percent
  - notes section:
    - whether K-Fold CV was used (+ fold count if used)
    - Path 2 class fixed as NLOS per brief
    - Path 2 target derived from second CIR peak offset
    - model uses raw features
    - physical constraint enforced by clipping `pred_p2 >= pred_p1`
    - `PEAK2_GAP = PEAK2_IDX - FP_IDX` rationale tied to brief hint

---

## 8) Final Console Summary
- Print completion banner.
- Print final Path 1 and Path 2 metrics.
- Print output directory and list of produced artifacts:
  - `xgb_regression_model_comparison.csv`
  - `xgb_range_estimation_results.png`
  - `xgb_regression_metrics_comparison.png`
  - `xgb_regressor_feature_importance.png`
  - `xgb_fp_idx_vs_range.png`
  - `xgb_peak2_gap_vs_extra_range.png`
  - `xgb_residual_distribution.png`
  - `xgb_p1_vs_p2_predicted.png`
  - `xgb_regression_results.txt`
  - `xgb_range_path1.pkl`, `xgb_range_path2.pkl`
  - `y_p1_pred_xgb.npy`, `y_p2_pred_xgb.npy`

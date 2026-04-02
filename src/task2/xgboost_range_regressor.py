"""
UWB LOS/NLOS - Range Estimator (XGBoost Only)
==============================================
Standalone XGBoost regressor pipeline extracted from range_regressor.py.

Run second_path_features.py FIRST to generate the input .npy files.
"""

from pathlib import Path
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate

warnings.filterwarnings("ignore")

from xgboost import XGBRegressor


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "outputs" / "preprocessed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "range_regressor"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = OUTPUT_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
RUN_KFOLD = os.getenv("RUN_KFOLD", "0") == "1"
CV_SPLITS = int(os.getenv("CV_SPLITS", "5"))

print("=" * 80)
print("XGBOOST RANGE ESTIMATOR")
print("=" * 80)
print(f"K-Fold CV enabled: {RUN_KFOLD}")
if RUN_KFOLD:
    print(f"CV config: {CV_SPLITS}-fold, shuffle=True, random_state={RANDOM_SEED}")
print()


# -----------------------------------------------------------------------------
# STEP 1: LOAD DATA
# -----------------------------------------------------------------------------
print("Step 1: Loading regression data...")

X_train = np.load(DATA_DIR / "X_train_regression.npy")
X_test = np.load(DATA_DIR / "X_test_regression.npy")
y_p1_train = np.load(DATA_DIR / "y_range_p1_train.npy")
y_p1_test = np.load(DATA_DIR / "y_range_p1_test.npy")
y_p2_train = np.load(DATA_DIR / "y_range_p2_train.npy")
y_p2_test = np.load(DATA_DIR / "y_range_p2_test.npy")

print(f"  Training samples : {len(X_train):,}")
print(f"  Test samples     : {len(X_test):,}")
print(f"  Features         : {X_train.shape[1]}")
print(f"  Path 1 range     : {y_p1_train.min():.2f} - {y_p1_train.max():.2f} m")
print(f"  Path 2 range     : {y_p2_train.min():.2f} - {y_p2_train.max():.2f} m")
print()


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def evaluate_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def to_float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def build_xgb_regressor():
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=1200,
        max_depth=8,
        learning_rate=0.015,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_alpha=0.0,
        reg_lambda=2.0,
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )


def fit_and_evaluate(path_label, model, X_tr, y_tr, X_te, y_te):
    print(f"  XGBoost - {path_label}")
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    t1 = time.time()
    y_pred = model.predict(X_te)
    predict_time = time.time() - t1

    rmse, mae, r2 = evaluate_metrics(y_te, y_pred)

    print(f"    Train time : {train_time:.2f}s")
    print(f"    RMSE       : {rmse:.4f} m")
    print(f"    MAE        : {mae:.4f} m")
    print(f"    R2         : {r2:.4f}")

    return {
        "model": model,
        "y_pred": y_pred,
        "train_time": train_time,
        "predict_time": predict_time,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def run_kfold_cv(path_label, X_tr, y_tr):
    print(f"  XGBoost - {path_label} (K-Fold CV)")
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    cv_scores = cross_validate(
        build_xgb_regressor(),
        X_tr,
        y_tr,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    rmse_vals = -cv_scores["test_rmse"]
    mae_vals = -cv_scores["test_mae"]
    r2_vals = cv_scores["test_r2"]

    summary = {
        "cv_rmse_mean": float(np.mean(rmse_vals)),
        "cv_rmse_std": float(np.std(rmse_vals)),
        "cv_mae_mean": float(np.mean(mae_vals)),
        "cv_mae_std": float(np.std(mae_vals)),
        "cv_r2_mean": float(np.mean(r2_vals)),
        "cv_r2_std": float(np.std(r2_vals)),
    }

    print(
        f"    CV RMSE    : {summary['cv_rmse_mean']:.4f} +/- {summary['cv_rmse_std']:.4f} m"
    )
    print(
        f"    CV MAE     : {summary['cv_mae_mean']:.4f} +/- {summary['cv_mae_std']:.4f} m"
    )
    print(
        f"    CV R2      : {summary['cv_r2_mean']:.4f} +/- {summary['cv_r2_std']:.4f}"
    )

    return summary


# -----------------------------------------------------------------------------
# STEP 2: TRAIN AND EVALUATE
# -----------------------------------------------------------------------------
print("Step 2: Training and evaluating XGBoost...")
print("-" * 80)

cv_p1 = None
cv_p2 = None
if RUN_KFOLD:
    cv_p1 = run_kfold_cv(path_label="Path 1", X_tr=X_train, y_tr=y_p1_train)
    cv_p2 = run_kfold_cv(path_label="Path 2", X_tr=X_train, y_tr=y_p2_train)

p1_result = fit_and_evaluate(
    path_label="Path 1",
    model=build_xgb_regressor(),
    X_tr=X_train,
    y_tr=y_p1_train,
    X_te=X_test,
    y_te=y_p1_test,
)
p2_result = fit_and_evaluate(
    path_label="Path 2",
    model=build_xgb_regressor(),
    X_tr=X_train,
    y_tr=y_p2_train,
    X_te=X_test,
    y_te=y_p2_test,
)

results = {
    "p1": p1_result,
    "p2": p2_result,
}

results_df = pd.DataFrame(
    [
        {
            "model": "XGBoost",
            "model_key": "xgb",
            "path": "Path 1",
            "rmse": p1_result["rmse"],
            "mae": p1_result["mae"],
            "r2": p1_result["r2"],
            "train_seconds": p1_result["train_time"],
            "predict_seconds": p1_result["predict_time"],
            "cv_used": RUN_KFOLD,
            "cv_splits": CV_SPLITS if RUN_KFOLD else np.nan,
            "cv_rmse_mean": cv_p1["cv_rmse_mean"] if cv_p1 else np.nan,
            "cv_rmse_std": cv_p1["cv_rmse_std"] if cv_p1 else np.nan,
            "cv_mae_mean": cv_p1["cv_mae_mean"] if cv_p1 else np.nan,
            "cv_mae_std": cv_p1["cv_mae_std"] if cv_p1 else np.nan,
            "cv_r2_mean": cv_p1["cv_r2_mean"] if cv_p1 else np.nan,
            "cv_r2_std": cv_p1["cv_r2_std"] if cv_p1 else np.nan,
        },
        {
            "model": "XGBoost",
            "model_key": "xgb",
            "path": "Path 2",
            "rmse": p2_result["rmse"],
            "mae": p2_result["mae"],
            "r2": p2_result["r2"],
            "train_seconds": p2_result["train_time"],
            "predict_seconds": p2_result["predict_time"],
            "cv_used": RUN_KFOLD,
            "cv_splits": CV_SPLITS if RUN_KFOLD else np.nan,
            "cv_rmse_mean": cv_p2["cv_rmse_mean"] if cv_p2 else np.nan,
            "cv_rmse_std": cv_p2["cv_rmse_std"] if cv_p2 else np.nan,
            "cv_mae_mean": cv_p2["cv_mae_mean"] if cv_p2 else np.nan,
            "cv_mae_std": cv_p2["cv_mae_std"] if cv_p2 else np.nan,
            "cv_r2_mean": cv_p2["cv_r2_mean"] if cv_p2 else np.nan,
            "cv_r2_std": cv_p2["cv_r2_std"] if cv_p2 else np.nan,
        },
    ]
)

comparison_csv = OUTPUT_DIR / "xgb_regression_model_comparison.csv"
results_df.to_csv(comparison_csv, index=False)
print(f"\n  Saved: {comparison_csv.name}")


# -----------------------------------------------------------------------------
# PHYSICAL CONSTRAINT: path 2 range must be >= path 1 range.
# -----------------------------------------------------------------------------
print("\nEnforcing physical constraint: pred_p2 >= pred_p1 ...")
p1_pred = results["p1"]["y_pred"]
p2_pred = results["p2"]["y_pred"]
violations_before = int((p2_pred < p1_pred).sum())
p2_clipped = np.maximum(p2_pred, p1_pred)
results["p2"]["y_pred"] = p2_clipped

if violations_before > 0:
    rmse_new, mae_new, r2_new = evaluate_metrics(y_p2_test, p2_clipped)
    results["p2"]["rmse"] = rmse_new
    results["p2"]["mae"] = mae_new
    results["p2"]["r2"] = r2_new

    mask = results_df["path"] == "Path 2"
    results_df.loc[mask, "rmse"] = rmse_new
    results_df.loc[mask, "mae"] = mae_new
    results_df.loc[mask, "r2"] = r2_new
    print(
        f"  XGBoost      - clipped {violations_before:,} violations (new Path2 RMSE={rmse_new:.4f}m)"
    )
else:
    print("  XGBoost      - no violations (constraint already satisfied)")

results_df.to_csv(comparison_csv, index=False)
print(f"  Updated: {comparison_csv.name}")


# -----------------------------------------------------------------------------
# STEP 3: SAVE NOTEBOOK-FRIENDLY ARTIFACTS
# -----------------------------------------------------------------------------
print("\nStep 3: Saving notebook-friendly artifacts...")

np.save(ARTIFACT_DIR / "y_true_p1.npy", y_p1_test)
np.save(ARTIFACT_DIR / "y_pred_p1.npy", results["p1"]["y_pred"])
np.save(ARTIFACT_DIR / "y_true_p2.npy", y_p2_test)
np.save(ARTIFACT_DIR / "y_pred_p2.npy", results["p2"]["y_pred"])
print("  Saved: artifacts/y_true_p1.npy")
print("  Saved: artifacts/y_pred_p1.npy")
print("  Saved: artifacts/y_true_p2.npy")
print("  Saved: artifacts/y_pred_p2.npy")


# Feature importance for XGBoost
print("  Saving feature importance for XGBoost...")

# Feature names must match second_path_features.py's core_features list (no RANGE).
core_feats = [
    "FP_IDX",
    "FP_AMP1",
    "FP_AMP2",
    "FP_AMP3",
    "STDEV_NOISE",
    "CIR_PWR",
    "MAX_NOISE",
    "RXPACC",
    "CH",
    "FRAME_LEN",
    "PREAM_LEN",
    "BITRATE",
    "PRFR",
    "SNR",
    "SNR_dB",
    "PEAK2_IDX",
    "PEAK2_AMP",
    "PEAK2_GAP",
    "PEAK2_FOUND",
    "FP_AMP_ratio",
    "SNR_per_acc",
    "signal_to_noise",
    "noise_ratio",
    "FP_power",
    "CIR_energy",
    "CIR_kurtosis",
    "CIR_skewness",
    "CIR_rise_time",
    "CIR_num_peaks",
]
cir_feats = [f"CIR_aligned_{i}" for i in range(120)]
feat_names = core_feats + cir_feats
if len(feat_names) != X_train.shape[1]:
    feat_names = [f"f{i}" for i in range(X_train.shape[1])]

for suffix, model_obj in [
    ("p1", results["p1"]["model"]),
    ("p2", results["p2"]["model"]),
]:
    importances = model_obj.feature_importances_
    pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values(
        "Importance", ascending=False
    ).reset_index(drop=True).to_csv(
        ARTIFACT_DIR / f"feature_importance_{suffix}.csv", index=False
    )
    print(f"  Saved: artifacts/feature_importance_{suffix}.csv")


# Load pair labels for report summary.
pair_labels_path = DATA_DIR / "y_test_pair.npy"
if pair_labels_path.exists():
    y_pair_test = np.load(pair_labels_path)
else:
    y_pair_test = np.load(DATA_DIR / "y_test.npy")

los_nlos_count = int(np.sum(y_pair_test == 0))
nlos_nlos_count = int(np.sum(y_pair_test == 1))
total = len(y_pair_test)


# -----------------------------------------------------------------------------
# STEP 4: SAVE RESULTS REPORT
# -----------------------------------------------------------------------------
print("\nStep 4: Saving results report...")

report_path = OUTPUT_DIR / "xgb_regression_results.txt"
with open(report_path, "w") as f:
    f.write("XGBOOST RANGE ESTIMATION RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Training samples : {len(X_train):,}\n")
    f.write(f"Test samples     : {len(X_test):,}\n")
    f.write(f"Features         : {X_train.shape[1]}\n\n")

    for path_name in ["Path 1", "Path 2"]:
        row = results_df[results_df["path"] == path_name].iloc[0]
        f.write(f"{path_name.upper()} PERFORMANCE\n")
        f.write(
            f"  XGBoost       RMSE={row['rmse']:.4f} m  MAE={row['mae']:.4f} m  "
            f"R2={row['r2']:.4f}  Train={row['train_seconds']:.2f}s\n"
        )

        cv_rmse_mean = to_float(row.get("cv_rmse_mean", np.nan))
        cv_rmse_std = to_float(row.get("cv_rmse_std", np.nan))
        cv_mae_mean = to_float(row.get("cv_mae_mean", np.nan))
        cv_mae_std = to_float(row.get("cv_mae_std", np.nan))
        cv_r2_mean = to_float(row.get("cv_r2_mean", np.nan))
        cv_r2_std = to_float(row.get("cv_r2_std", np.nan))
        cv_splits = to_int(row.get("cv_splits", CV_SPLITS), default=CV_SPLITS)

        if RUN_KFOLD and np.isfinite(cv_rmse_mean):
            f.write(
                f"    CV({cv_splits}-fold): "
                f"RMSE={cv_rmse_mean:.4f}+/-{cv_rmse_std:.4f} m  "
                f"MAE={cv_mae_mean:.4f}+/-{cv_mae_std:.4f} m  "
                f"R2={cv_r2_mean:.4f}+/-{cv_r2_std:.4f}\n"
            )
        f.write("\n")

    f.write("TWO-PATH PAIR CLASSIFICATION (test split)\n")
    f.write(
        f"  LOS + NLOS  pairs : {los_nlos_count:,} ({los_nlos_count / total * 100:.1f}%)\n"
    )
    f.write(
        f"  NLOS + NLOS pairs : {nlos_nlos_count:,} ({nlos_nlos_count / total * 100:.1f}%)\n\n"
    )

    f.write("NOTES\n")
    f.write(f"  - K-Fold CV used for model comparison on training split: {RUN_KFOLD}")
    if RUN_KFOLD:
        f.write(f" ({CV_SPLITS}-fold).\n")
    else:
        f.write(".\n")
    f.write("  - Path 2 class is fixed to NLOS per project brief.\n")
    f.write("  - Path 2 target range is derived from second CIR peak offset.\n")
    f.write("  - XGBoost uses raw features.\n")
    f.write("  - Physical constraint enforced: pred_p2 clipped to >= pred_p1.\n")
    f.write("  - PEAK2_GAP = PEAK2_IDX - FP_IDX encodes the brief hint on using\n")
    f.write("    FP_IDX and range to correlate to the second dominant path.\n")

print(f"  Saved: {report_path.name}")


# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
print()
print("=" * 80)
print("XGBOOST RANGE ESTIMATION COMPLETE")
print("=" * 80)
print(
    f"\nPath 1: RMSE={results['p1']['rmse']:.4f}m  MAE={results['p1']['mae']:.4f}m  "
    f"R2={results['p1']['r2']:.4f}"
)
print(
    f"Path 2: RMSE={results['p2']['rmse']:.4f}m  MAE={results['p2']['mae']:.4f}m  "
    f"R2={results['p2']['r2']:.4f}"
)
print("\nOutput files in:", OUTPUT_DIR.absolute())
print("  - xgb_regression_model_comparison.csv")
print("  - xgb_regression_results.txt")
print("  - artifacts/y_true_p1.npy")
print("  - artifacts/y_pred_p1.npy")
print("  - artifacts/y_true_p2.npy")
print("  - artifacts/y_pred_p2.npy")
print("  - artifacts/feature_importance_p1.csv")
print("  - artifacts/feature_importance_p2.csv")
print("Use notebooks/range_regressor_analysis.ipynb to regenerate plots.")
print("=" * 80)

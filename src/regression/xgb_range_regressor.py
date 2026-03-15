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

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
except Exception as exc:
    raise RuntimeError(
        "xgboost is required for xgb_range_regressor.py. "
        f"Install it first. Import error: {exc}"
    ) from exc


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
RUN_NAME = os.getenv("RUN_NAME", "split_env_70_15_15_seed42")
DATA_DIR = Path("./runs") / RUN_NAME / "preprocessed_data"
OUTPUT_DIR = Path("./runs") / RUN_NAME / "models" / "range_regressor"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
XGB_TREE_METHOD = os.getenv("XGB_TREE_METHOD", "hist")
RUN_KFOLD = os.getenv("RUN_KFOLD", "0") == "1"
CV_SPLITS = int(os.getenv("CV_SPLITS", "5"))
CV_N_JOBS = int(os.getenv("CV_N_JOBS", "-1"))

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
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method=XGB_TREE_METHOD,
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
        n_jobs=CV_N_JOBS,
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
# STEP 3: VISUALIZATIONS
# -----------------------------------------------------------------------------
print("\nStep 3: Generating visualizations...")
plt.style.use("seaborn-v0_8-darkgrid")

# 3A. Predicted vs actual scatter
fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)

ax = axes[0, 0]
ax.scatter(y_p1_test, results["p1"]["y_pred"], alpha=0.25, s=8, color="#2d7dd2")
lims = [
    min(y_p1_test.min(), results["p1"]["y_pred"].min()),
    max(y_p1_test.max(), results["p1"]["y_pred"].max()),
]
ax.plot(lims, lims, "k--", linewidth=1.5)
ax.set_xlabel("Actual Range (m)")
ax.set_ylabel("Predicted Range (m)")
ax.set_title(
    f"Path 1 - XGBoost\nRMSE={results['p1']['rmse']:.3f}  MAE={results['p1']['mae']:.3f}  R2={results['p1']['r2']:.3f}",
    fontsize=11,
    fontweight="bold",
)

ax = axes[0, 1]
ax.scatter(y_p2_test, results["p2"]["y_pred"], alpha=0.25, s=8, color="#d1495b")
lims = [
    min(y_p2_test.min(), results["p2"]["y_pred"].min()),
    max(y_p2_test.max(), results["p2"]["y_pred"].max()),
]
ax.plot(lims, lims, "k--", linewidth=1.5)
ax.set_xlabel("Actual Range (m)")
ax.set_ylabel("Predicted Range (m)")
ax.set_title(
    f"Path 2 - XGBoost\nRMSE={results['p2']['rmse']:.3f}  MAE={results['p2']['mae']:.3f}  R2={results['p2']['r2']:.3f}",
    fontsize=11,
    fontweight="bold",
)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "xgb_range_estimation_results.png", dpi=300, bbox_inches="tight"
)
plt.close()
print("  Saved: xgb_range_estimation_results.png")


# 3B. Metrics comparison bars
metrics = ["rmse", "mae", "r2"]
metric_titles = {"rmse": "RMSE (m)", "mae": "MAE (m)", "r2": "R2"}
fig, axes = plt.subplots(2, 3, figsize=(16, 8), squeeze=False)

for row_idx, path_name in enumerate(["Path 1", "Path 2"]):
    path_df = results_df[results_df["path"] == path_name]
    for col_idx, metric in enumerate(metrics):
        ax = axes[row_idx, col_idx]
        val = float(path_df[metric].iloc[0])
        bar = ax.bar(["XGBoost"], [val], color="#3bb273", alpha=0.9)[0]
        ax.set_title(f"{path_name} - {metric_titles[metric]}", fontweight="bold")
        ax.set_ylabel(metric_titles[metric])
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "xgb_regression_metrics_comparison.png", dpi=300, bbox_inches="tight"
)
plt.close()
print("  Saved: xgb_regression_metrics_comparison.png")


# 3C. Feature importance for XGBoost
print("  Plotting feature importance for XGBoost...")

core_feats = [
    "RANGE",
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
]
cir_feats = [f"CIR{i}" for i in range(730, 850)]
feat_names = core_feats + cir_feats
if len(feat_names) != X_train.shape[1]:
    feat_names = [f"f{i}" for i in range(X_train.shape[1])]

fig, axes = plt.subplots(1, 2, figsize=(18, 6), squeeze=False)
top_n = 20

for col_idx, (path_label, model_obj) in enumerate(
    [("Path 1", results["p1"]["model"]), ("Path 2", results["p2"]["model"])]
):
    ax = axes[0, col_idx]
    importances = model_obj.feature_importances_
    top_idx = np.argsort(importances)[::-1][:top_n]
    y_pos = np.arange(top_n)
    ax.barh(y_pos, importances[top_idx], color="#577590", alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feat_names[i] for i in top_idx], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"XGBoost - {path_label} Top {top_n}", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "xgb_regressor_feature_importance.png", dpi=300, bbox_inches="tight"
)
plt.close()
print("  Saved: xgb_regressor_feature_importance.png")


# 3D. Two-path pair classification summary
pair_labels_path = DATA_DIR / "y_test_pair.npy"
if pair_labels_path.exists():
    y_p1_class = np.load(pair_labels_path)
else:
    y_p1_class = np.load(DATA_DIR / "y_test.npy")

los_nlos_count = int(np.sum(y_p1_class == 0))
nlos_nlos_count = int(np.sum(y_p1_class == 1))
total = len(y_p1_class)

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    ["LOS + NLOS\n(Path1=LOS)", "NLOS + NLOS\n(Path1=NLOS)"],
    [los_nlos_count, nlos_nlos_count],
    color=["#2ecc71", "#e74c3c"],
    alpha=0.85,
    edgecolor="white",
)
for bar, cnt in zip(bars, [los_nlos_count, nlos_nlos_count]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{cnt:,}\n({cnt / total * 100:.1f}%)",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=11,
    )
ax.set_ylabel("Number of Samples")
ax.set_title(
    "Two-Path Pair Classification\n(Path 2 is always NLOS per brief)",
    fontweight="bold",
    fontsize=13,
)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "xgb_two_path_pair_distribution.png", dpi=300, bbox_inches="tight"
)
plt.close()
print("  Saved: xgb_two_path_pair_distribution.png")


# -----------------------------------------------------------------------------
# STEP 4: SAVE MODELS, PREDICTIONS, AND REPORT
# -----------------------------------------------------------------------------
print("\nStep 4: Saving models and results...")

model_p1_path = OUTPUT_DIR / "xgb_range_path1.pkl"
model_p2_path = OUTPUT_DIR / "xgb_range_path2.pkl"
pred_p1_path = OUTPUT_DIR / "y_p1_pred_xgb.npy"
pred_p2_path = OUTPUT_DIR / "y_p2_pred_xgb.npy"

joblib.dump(results["p1"]["model"], model_p1_path)
joblib.dump(results["p2"]["model"], model_p2_path)
np.save(pred_p1_path, results["p1"]["y_pred"])
np.save(pred_p2_path, results["p2"]["y_pred"])

print(f"  Saved: {model_p1_path.name}, {model_p2_path.name}")
print(f"  Saved: {pred_p1_path.name}, {pred_p2_path.name}")

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

    f.write("TWO-PATH PAIR CLASSIFICATION\n")
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
print("  - xgb_range_estimation_results.png")
print("  - xgb_regression_metrics_comparison.png")
print("  - xgb_regressor_feature_importance.png")
print("  - xgb_two_path_pair_distribution.png")
print("  - xgb_regression_results.txt")
print("  - xgb_range_path1.pkl / xgb_range_path2.pkl")
print("  - y_p1_pred_xgb.npy / y_p2_pred_xgb.npy")
print("=" * 80)

"""
UWB LOS/NLOS - Range Estimator (Regression)
============================================
Part 2 of the project brief:
  "Predict the measured range for the two dominant shortest paths"

Models trained in this script:
  1) RandomForestRegressor
  2) KNeighborsRegressor
  3) XGBRegressor (if xgboost is installed)

Run second_path_features.py FIRST to generate the input .npy files.
"""

from pathlib import Path
import math
import os
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
    XGB_IMPORT_ERROR = ""
except Exception as exc:
    HAS_XGBOOST = False
    XGB_IMPORT_ERROR = str(exc)
    XGBRegressor = None  # type: ignore[assignment]


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
print("MULTI-MODEL RANGE ESTIMATOR (RF + KNN + XGBOOST)")
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


# KNN is distance-based and needs scaling. Tree models can use raw values.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


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


def fit_and_evaluate(model_name, path_label, model, X_tr, y_tr, X_te, y_te):
    print(f"  {model_name} - {path_label}")
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


def run_kfold_cv(model_name, path_label, model_builder, use_scaled, X_tr_raw, y_tr):
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    if use_scaled:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", model_builder()),
            ]
        )
    else:
        model = model_builder()

    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    cv_scores = cross_validate(
        model,
        X_tr_raw,
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
        f"    CV RMSE    : {summary['cv_rmse_mean']:.4f} ± {summary['cv_rmse_std']:.4f} m"
    )
    print(
        f"    CV MAE     : {summary['cv_mae_mean']:.4f} ± {summary['cv_mae_std']:.4f} m"
    )
    print(f"    CV R2      : {summary['cv_r2_mean']:.4f} ± {summary['cv_r2_std']:.4f}")

    return summary


def build_model_specs():
    specs = [
        {
            "key": "rf",
            "name": "RandomForest",
            "use_scaled": False,
            "builder": lambda: RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=0,
            ),
        },
        {
            "key": "knn",
            "name": "KNN",
            "use_scaled": True,
            "builder": lambda: KNeighborsRegressor(
                n_neighbors=15,
                weights="distance",
                p=2,
                n_jobs=-1,
            ),
        },
    ]

    xgb_regressor_cls = XGBRegressor
    if HAS_XGBOOST and xgb_regressor_cls is not None:
        specs.append(
            {
                "key": "xgb",
                "name": "XGBoost",
                "use_scaled": False,
                "builder": lambda cls=xgb_regressor_cls: cls(
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
                ),
            }
        )
    else:
        print("Warning: xgboost not available, skipping XGBoost regressor.")
        print(f"  Import error: {XGB_IMPORT_ERROR}")
        print()

    return specs


def summarize_best_models(results_df):
    best = {}
    for path_name in sorted(results_df["path"].unique()):
        path_rows = results_df[results_df["path"] == path_name]
        best_row = path_rows.nsmallest(1, columns="rmse").iloc[0]
        best[path_name] = {
            "model": best_row["model"],
            "rmse": float(best_row["rmse"]),
            "mae": float(best_row["mae"]),
            "r2": float(best_row["r2"]),
        }
    return best


# -----------------------------------------------------------------------------
# STEP 2: TRAIN AND EVALUATE ALL MODELS
# -----------------------------------------------------------------------------
print("Step 2: Training and evaluating models...")
print("-" * 80)

model_specs = build_model_specs()

results = {}
rows = []

for spec in model_specs:
    model_key = spec["key"]
    model_name = spec["name"]
    use_scaled = spec["use_scaled"]

    X_tr = X_train_scaled if use_scaled else X_train
    X_te = X_test_scaled if use_scaled else X_test

    print(f"\nTraining {model_name} ({'scaled input' if use_scaled else 'raw input'})")

    cv_p1 = None
    cv_p2 = None
    if RUN_KFOLD:
        print(f"  {model_name} - Path 1 (K-Fold CV)")
        cv_p1 = run_kfold_cv(
            model_name=model_name,
            path_label="Path 1",
            model_builder=spec["builder"],
            use_scaled=use_scaled,
            X_tr_raw=X_train,
            y_tr=y_p1_train,
        )
        print(f"  {model_name} - Path 2 (K-Fold CV)")
        cv_p2 = run_kfold_cv(
            model_name=model_name,
            path_label="Path 2",
            model_builder=spec["builder"],
            use_scaled=use_scaled,
            X_tr_raw=X_train,
            y_tr=y_p2_train,
        )

    p1_result = fit_and_evaluate(
        model_name=model_name,
        path_label="Path 1",
        model=spec["builder"](),
        X_tr=X_tr,
        y_tr=y_p1_train,
        X_te=X_te,
        y_te=y_p1_test,
    )
    p2_result = fit_and_evaluate(
        model_name=model_name,
        path_label="Path 2",
        model=spec["builder"](),
        X_tr=X_tr,
        y_tr=y_p2_train,
        X_te=X_te,
        y_te=y_p2_test,
    )

    results[(model_key, "p1")] = p1_result
    results[(model_key, "p2")] = p2_result

    rows.append(
        {
            "model": model_name,
            "model_key": model_key,
            "path": "Path 1",
            "rmse": p1_result["rmse"],
            "mae": p1_result["mae"],
            "r2": p1_result["r2"],
            "train_seconds": p1_result["train_time"],
            "predict_seconds": p1_result["predict_time"],
            "scaled_input": use_scaled,
            "cv_used": RUN_KFOLD,
            "cv_splits": CV_SPLITS if RUN_KFOLD else np.nan,
            "cv_rmse_mean": cv_p1["cv_rmse_mean"] if cv_p1 else np.nan,
            "cv_rmse_std": cv_p1["cv_rmse_std"] if cv_p1 else np.nan,
            "cv_mae_mean": cv_p1["cv_mae_mean"] if cv_p1 else np.nan,
            "cv_mae_std": cv_p1["cv_mae_std"] if cv_p1 else np.nan,
            "cv_r2_mean": cv_p1["cv_r2_mean"] if cv_p1 else np.nan,
            "cv_r2_std": cv_p1["cv_r2_std"] if cv_p1 else np.nan,
        }
    )
    rows.append(
        {
            "model": model_name,
            "model_key": model_key,
            "path": "Path 2",
            "rmse": p2_result["rmse"],
            "mae": p2_result["mae"],
            "r2": p2_result["r2"],
            "train_seconds": p2_result["train_time"],
            "predict_seconds": p2_result["predict_time"],
            "scaled_input": use_scaled,
            "cv_used": RUN_KFOLD,
            "cv_splits": CV_SPLITS if RUN_KFOLD else np.nan,
            "cv_rmse_mean": cv_p2["cv_rmse_mean"] if cv_p2 else np.nan,
            "cv_rmse_std": cv_p2["cv_rmse_std"] if cv_p2 else np.nan,
            "cv_mae_mean": cv_p2["cv_mae_mean"] if cv_p2 else np.nan,
            "cv_mae_std": cv_p2["cv_mae_std"] if cv_p2 else np.nan,
            "cv_r2_mean": cv_p2["cv_r2_mean"] if cv_p2 else np.nan,
            "cv_r2_std": cv_p2["cv_r2_std"] if cv_p2 else np.nan,
        }
    )

results_df = pd.DataFrame(rows)
results_df = results_df.sort_values(
    ["path", "rmse", "mae"], ascending=[True, True, True]
)
best_by_path = summarize_best_models(results_df)

comparison_csv = OUTPUT_DIR / "regression_model_comparison.csv"
results_df.to_csv(comparison_csv, index=False)
print(f"\n  Saved: {comparison_csv.name}")


# -----------------------------------------------------------------------------
# PHYSICAL CONSTRAINT: path 2 range must be >= path 1 range.
# Path 1 is always the shortest path; path 2 is a secondary (longer) path.
# Enforce this by clipping each model's path 2 predictions from below.
# -----------------------------------------------------------------------------
print("\nEnforcing physical constraint: pred_p2 >= pred_p1 ...")
for spec in model_specs:
    key = spec["key"]
    p1_pred = results[(key, "p1")]["y_pred"]
    p2_pred = results[(key, "p2")]["y_pred"]

    violations_before = int((p2_pred < p1_pred).sum())
    p2_clipped = np.maximum(p2_pred, p1_pred)
    results[(key, "p2")]["y_pred"] = p2_clipped

    if violations_before > 0:
        # Recompute metrics after clipping
        rmse_new, mae_new, r2_new = evaluate_metrics(y_p2_test, p2_clipped)
        results[(key, "p2")]["rmse"] = rmse_new
        results[(key, "p2")]["mae"] = mae_new
        results[(key, "p2")]["r2"] = r2_new
        # Update results_df row
        mask = (results_df["model_key"] == key) & (results_df["path"] == "Path 2")
        results_df.loc[mask, "rmse"] = rmse_new
        results_df.loc[mask, "mae"] = mae_new
        results_df.loc[mask, "r2"] = r2_new
        print(
            f"  {spec['name']:13s} — clipped {violations_before:,} violations "
            f"(new Path2 RMSE={rmse_new:.4f}m)"
        )
    else:
        print(f"  {spec['name']:13s} — no violations (constraint already satisfied)")

# Re-sort and save updated comparison after clipping
results_df = results_df.sort_values(
    ["path", "rmse", "mae"], ascending=[True, True, True]
)
best_by_path = summarize_best_models(results_df)
results_df.to_csv(comparison_csv, index=False)
print(f"  Updated: {comparison_csv.name}")


# -----------------------------------------------------------------------------
# STEP 3: VISUALIZATIONS
# -----------------------------------------------------------------------------
print("\nStep 3: Generating visualizations...")
plt.style.use("seaborn-v0_8-darkgrid")

model_order = [spec["key"] for spec in model_specs]
model_name_map = {spec["key"]: spec["name"] for spec in model_specs}


# 3A. Predicted vs actual scatter for each model/path
n_models = len(model_order)
fig, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 10), squeeze=False)

for col, model_key in enumerate(model_order):
    model_name = model_name_map[model_key]

    p1 = results[(model_key, "p1")]
    ax = axes[0, col]
    ax.scatter(y_p1_test, p1["y_pred"], alpha=0.25, s=8, color="#2d7dd2")
    lims = [
        min(y_p1_test.min(), p1["y_pred"].min()),
        max(y_p1_test.max(), p1["y_pred"].max()),
    ]
    ax.plot(lims, lims, "k--", linewidth=1.5)
    ax.set_xlabel("Actual Range (m)")
    ax.set_ylabel("Predicted Range (m)")
    ax.set_title(
        f"Path 1 - {model_name}\nRMSE={p1['rmse']:.3f}  MAE={p1['mae']:.3f}  R2={p1['r2']:.3f}",
        fontsize=11,
        fontweight="bold",
    )

    p2 = results[(model_key, "p2")]
    ax = axes[1, col]
    ax.scatter(y_p2_test, p2["y_pred"], alpha=0.25, s=8, color="#d1495b")
    lims = [
        min(y_p2_test.min(), p2["y_pred"].min()),
        max(y_p2_test.max(), p2["y_pred"].max()),
    ]
    ax.plot(lims, lims, "k--", linewidth=1.5)
    ax.set_xlabel("Actual Range (m)")
    ax.set_ylabel("Predicted Range (m)")
    ax.set_title(
        f"Path 2 - {model_name}\nRMSE={p2['rmse']:.3f}  MAE={p2['mae']:.3f}  R2={p2['r2']:.3f}",
        fontsize=11,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "range_estimation_results.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: range_estimation_results.png")


# 3B. Metrics comparison bars
metrics = ["rmse", "mae", "r2"]
metric_titles = {"rmse": "RMSE (m)", "mae": "MAE (m)", "r2": "R2"}
fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)

for row_idx, path_name in enumerate(["Path 1", "Path 2"]):
    path_df = results_df[results_df["path"] == path_name].copy()
    path_df["model"] = pd.Categorical(
        path_df["model"],
        categories=[model_name_map[m] for m in model_order],
        ordered=True,
    )
    model_labels = np.asarray(path_df["model"].astype(str))
    path_df = path_df.iloc[np.argsort(model_labels)]

    for col_idx, metric in enumerate(metrics):
        ax = axes[row_idx, col_idx]
        vals = path_df[metric].to_numpy()
        labels = path_df["model"].astype(str).to_list()
        bars = ax.bar(
            labels,
            vals,
            color=["#2d7dd2", "#f4a259", "#3bb273"][: len(labels)],
            alpha=0.9,
        )
        ax.set_title(f"{path_name} - {metric_titles[metric]}", fontweight="bold")
        ax.set_ylabel(metric_titles[metric])
        ax.tick_params(axis="x", rotation=15)

        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "regression_metrics_comparison.png", dpi=300, bbox_inches="tight"
)
plt.close()
print("  Saved: regression_metrics_comparison.png")


# 3C. Feature importance for tree models (RF and XGBoost)
print("  Plotting feature importance for tree models...")

core_feats = [
    "RANGE",  # path 1 measured range — included in pair feature set
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
    # Second-path features — directly address the brief hint:
    # "Use FP_IDX and measured range to correlate to the second dominant path."
    # PEAK2_GAP = PEAK2_IDX - FP_IDX is the sample-domain time-of-flight delta
    # between the first and second dominant CIR peaks, encoding the extra path
    # length of path 2 relative to path 1.
    "PEAK2_IDX",
    "PEAK2_AMP",
    "PEAK2_GAP",  # = PEAK2_IDX - FP_IDX  (brief hint feature)
    "PEAK2_FOUND",
]
cir_feats = [f"CIR{i}" for i in range(730, 850)]
feat_names = core_feats + cir_feats

if len(feat_names) != X_train.shape[1]:
    feat_names = [f"f{i}" for i in range(X_train.shape[1])]

importance_items = []
for tree_key, tree_name in [("rf", "RandomForest"), ("xgb", "XGBoost")]:
    if (tree_key, "p1") in results:
        importance_items.append(
            (
                tree_name,
                "Path 1",
                results[(tree_key, "p1")]["model"].feature_importances_,
            )
        )
    if (tree_key, "p2") in results:
        importance_items.append(
            (
                tree_name,
                "Path 2",
                results[(tree_key, "p2")]["model"].feature_importances_,
            )
        )

if importance_items:
    top_n = 20
    n_items = len(importance_items)
    n_cols = 2
    n_rows = int(math.ceil(n_items / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows), squeeze=False)

    for idx, (model_name, path_name, importances) in enumerate(importance_items):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        top_idx = np.argsort(importances)[::-1][:top_n]
        y_pos = np.arange(top_n)
        ax.barh(y_pos, importances[top_idx], color="#577590", alpha=0.9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feat_names[i] for i in top_idx], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"{model_name} - {path_name} Top {top_n}", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    for idx in range(n_items, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "regressor_feature_importance.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("  Saved: regressor_feature_importance.png")


# 3D. Two-path pair classification summary (existing report artifact)
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
plt.savefig(OUTPUT_DIR / "two_path_pair_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: two_path_pair_distribution.png")


# -----------------------------------------------------------------------------
# STEP 4: SAVE MODELS, PREDICTIONS, AND REPORT
# -----------------------------------------------------------------------------
print("\nStep 4: Saving models and results...")

for spec in model_specs:
    key = spec["key"]

    p1_obj = results[(key, "p1")]
    p2_obj = results[(key, "p2")]

    model_p1_path = OUTPUT_DIR / f"{key}_range_path1.pkl"
    model_p2_path = OUTPUT_DIR / f"{key}_range_path2.pkl"
    pred_p1_path = OUTPUT_DIR / f"y_p1_pred_{key}.npy"
    pred_p2_path = OUTPUT_DIR / f"y_p2_pred_{key}.npy"

    joblib.dump(p1_obj["model"], model_p1_path)
    joblib.dump(p2_obj["model"], model_p2_path)
    np.save(pred_p1_path, p1_obj["y_pred"])
    np.save(pred_p2_path, p2_obj["y_pred"])

    print(f"  Saved: {model_p1_path.name}, {model_p2_path.name}")
    print(f"  Saved: {pred_p1_path.name}, {pred_p2_path.name}")

# Legacy compatibility with previous downstream consumers
if ("rf", "p1") in results and ("rf", "p2") in results:
    np.save(OUTPUT_DIR / "y_p1_pred.npy", results[("rf", "p1")]["y_pred"])
    np.save(OUTPUT_DIR / "y_p2_pred.npy", results[("rf", "p2")]["y_pred"])
    print("  Saved: y_p1_pred.npy, y_p2_pred.npy (RF legacy compatibility)")

report_path = OUTPUT_DIR / "regression_results.txt"
with open(report_path, "w") as f:
    f.write("RANGE ESTIMATION RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Training samples : {len(X_train):,}\n")
    f.write(f"Test samples     : {len(X_test):,}\n")
    f.write(f"Features         : {X_train.shape[1]}\n")
    f.write(f"Models trained   : {', '.join(results_df['model'].unique())}\n\n")

    for path_name in ["Path 1", "Path 2"]:
        f.write(f"{path_name.upper()} PERFORMANCE\n")
        path_rows = results_df[results_df["path"] == path_name].copy()
        path_rows = path_rows.iloc[np.argsort(np.asarray(path_rows["rmse"]))]
        for _, row in path_rows.iterrows():
            f.write(
                f"  {row['model']:<13} RMSE={row['rmse']:.4f} m  MAE={row['mae']:.4f} m  "
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
                    f"RMSE={cv_rmse_mean:.4f}±{cv_rmse_std:.4f} m  "
                    f"MAE={cv_mae_mean:.4f}±{cv_mae_std:.4f} m  "
                    f"R2={cv_r2_mean:.4f}±{cv_r2_std:.4f}\n"
                )
        best = best_by_path[path_name]
        f.write(
            f"  BEST ({path_name})  : {best['model']}  "
            f"(RMSE={best['rmse']:.4f} m, MAE={best['mae']:.4f} m, R2={best['r2']:.4f})\n\n"
        )

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
    f.write("  - KNN uses scaled features; RF and XGBoost use raw features.\n")
    f.write("  - Physical constraint enforced: pred_p2 clipped to >= pred_p1.\n")
    f.write("  - PEAK2_GAP = PEAK2_IDX - FP_IDX encodes the brief hint on using\n")
    f.write("    FP_IDX and range to correlate to the second dominant path.\n")

print(f"  Saved: {report_path.name}")


# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
print()
print("=" * 80)
print("RANGE ESTIMATION COMPLETE")
print("=" * 80)

print("\nBest model by path (lowest RMSE):")
for path_name in ["Path 1", "Path 2"]:
    best = best_by_path[path_name]
    print(
        f"  {path_name}: {best['model']}  "
        f"RMSE={best['rmse']:.4f}m  MAE={best['mae']:.4f}m  R2={best['r2']:.4f}"
    )

print("\nOutput files in:", OUTPUT_DIR.absolute())
print("  - regression_model_comparison.csv")
print("  - range_estimation_results.png")
print("  - regression_metrics_comparison.png")
print("  - regressor_feature_importance.png (tree models only)")
print("  - two_path_pair_distribution.png")
print("  - regression_results.txt")
print("  - *_range_path1.pkl / *_range_path2.pkl")
print("  - y_p1_pred_*.npy / y_p2_pred_*.npy")
print("=" * 80)

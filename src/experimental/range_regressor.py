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
import os
import time
import warnings

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
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "range_regressor_experimental"
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
# STEP 3: SAVE COMPARISON CSV ONLY
# -----------------------------------------------------------------------------
print("\nStep 3: Saving comparison results...")
print(f"  Saved: {comparison_csv.name}")


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
print("=" * 80)

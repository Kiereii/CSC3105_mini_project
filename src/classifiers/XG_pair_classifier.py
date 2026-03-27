"""
UWB LOS/NLOS - Pair-Level XGBoost Classifier
0 -> LOS+NLOS
1 -> NLOS+NLOS
"""

import os
import time
import warnings
import importlib
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")


RUN_NAME = os.getenv("RUN_NAME", "split_env_70_15_15_seed42")
DATA_DIR = Path("./runs") / RUN_NAME / "preprocessed_data"
OUTPUT_DIR = Path("./runs") / RUN_NAME / "models" / "pair_classifier"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
USE_GPU = os.getenv("USE_GPU", "0") == "1"
XGB_DEVICE = "cuda" if USE_GPU else "cpu"
XGB_TREE_METHOD = "hist"

RUN_TUNING = os.getenv("RUN_TUNING", "1") == "1"
PAIR_BAYES_TRIALS = int(os.getenv("PAIR_BAYES_TRIALS", "40"))
PAIR_CV_SPLITS = int(os.getenv("PAIR_CV_SPLITS", "3"))

N_ESTIMATORS = int(os.getenv("PAIR_XGB_N_ESTIMATORS", "200"))
MAX_DEPTH = int(os.getenv("PAIR_XGB_MAX_DEPTH", "6"))
LEARNING_RATE = float(os.getenv("PAIR_XGB_LEARNING_RATE", "0.1"))
SUBSAMPLE = float(os.getenv("PAIR_XGB_SUBSAMPLE", "0.8"))
COLSAMPLE_BYTREE = float(os.getenv("PAIR_XGB_COLSAMPLE_BYTREE", "0.8"))
MIN_CHILD_WEIGHT = int(os.getenv("PAIR_XGB_MIN_CHILD_WEIGHT", "1"))
GAMMA = float(os.getenv("PAIR_XGB_GAMMA", "0.0"))
REG_ALPHA = float(os.getenv("PAIR_XGB_REG_ALPHA", "0.0"))
REG_LAMBDA = float(os.getenv("PAIR_XGB_REG_LAMBDA", "1.0"))

CLASS_NAMES = ["LOS+NLOS", "NLOS+NLOS"]
ERROR_BOUNDARY_LOW = float(os.getenv("PAIR_ERROR_BOUNDARY_LOW", "0.4"))
ERROR_BOUNDARY_HIGH = float(os.getenv("PAIR_ERROR_BOUNDARY_HIGH", "0.6"))
CONFIDENT_ERROR_LOW = float(os.getenv("PAIR_CONFIDENT_ERROR_LOW", "0.2"))
CONFIDENT_ERROR_HIGH = float(os.getenv("PAIR_CONFIDENT_ERROR_HIGH", "0.8"))

print("=" * 80)
print("PAIR-LEVEL CLASSIFIER - XGBOOST ONLY")
print("=" * 80)
print(f"RUN_NAME : {RUN_NAME}")
print(f"Backend  : {'GPU (CUDA)' if USE_GPU else 'CPU'}")
print()
print("Pair label definition:")
print("  0 = LOS+NLOS  (trustworthy LOS path exists)")
print("  1 = NLOS+NLOS (both paths obstructed)")
print()


# ==============================================================================
# STEP 1: LOAD PAIR DATASET
# ==============================================================================
print("Step 1: Loading pair dataset...")

X_train = np.load(DATA_DIR / "X_train_pair.npy")
X_test = np.load(DATA_DIR / "X_test_pair.npy")
y_train = np.load(DATA_DIR / "y_train_pair.npy")
y_test = np.load(DATA_DIR / "y_test_pair.npy")


def load_pair_feature_names(path: Path, n_features: int) -> list[str]:
    """Parse pair_feature_names.txt and return a list of feature name strings.

    Falls back to generic names if the parsed count does not match the actual
    feature matrix width, preventing silent index misalignment in importance
    plots.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    names: list[str] = []
    reading_core = False
    cir_start: int | None = None
    cir_end: int | None = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Core + Second-Path Features"):
            reading_core = True
            continue
        if reading_core and stripped.startswith(tuple("0123456789")):
            names.append(line.split(".")[-1].strip())
        if stripped.startswith("CIR Features"):
            # Extract the explicit CIR range from the file rather than
            # hardcoding 730-850, so the names stay correct if preprocessing
            # ever changes the focus window.
            reading_core = False
            continue
        if stripped.startswith("Range: CIR"):
            # Expected format: "Range: CIR730 to CIR849"
            try:
                parts = stripped.replace("Range:", "").strip().split()
                cir_start = int(parts[0].replace("CIR", ""))
                cir_end = int(parts[2].replace("CIR", "")) + 1  # inclusive end
            except (IndexError, ValueError):
                cir_start, cir_end = 730, 850  # safe default
            for i in range(cir_start, cir_end):
                names.append(f"CIR{i}")
            break

    # If CIR block was never found, append a fallback range
    if cir_start is None:
        for i in range(730, 850):
            names.append(f"CIR{i}")

    # Guard: if parsed count disagrees with the actual feature matrix width,
    # fall back to generic names so importance plots are not silently wrong.
    if len(names) != n_features:
        print(
            f"  Warning: parsed {len(names)} feature names but matrix has "
            f"{n_features} columns. Falling back to generic names."
        )
        return [f"f{i}" for i in range(n_features)]

    return names


feature_names = load_pair_feature_names(
    DATA_DIR / "pair_feature_names.txt", X_train.shape[1]
)

print(f"  Training samples : {len(X_train):,}")
print(f"  Test samples     : {len(X_test):,}")
print(f"  Features         : {X_train.shape[1]}")
print()
print("  Pair label distribution:")
print(
    f"    Train LOS+NLOS  (0): {(y_train == 0).sum():,} "
    f"({(y_train == 0).mean() * 100:.1f}%)"
)
print(
    f"    Train NLOS+NLOS (1): {(y_train == 1).sum():,} "
    f"({(y_train == 1).mean() * 100:.1f}%)"
)
print(
    f"    Test  LOS+NLOS  (0): {(y_test == 0).sum():,} "
    f"({(y_test == 0).mean() * 100:.1f}%)"
)
print(
    f"    Test  NLOS+NLOS (1): {(y_test == 1).sum():,} "
    f"({(y_test == 1).mean() * 100:.1f}%)"
)
print()


# ==============================================================================
# STEP 2: OPTUNA TUNING + TRAIN XGBOOST MODEL
# ==============================================================================
print("Step 2: Tuning and training XGBoost pair classifier...")
print("-" * 80)

n_class0 = (y_train == 0).sum()  # LOS+NLOS (negative class)
n_class1 = (y_train == 1).sum()  # NLOS+NLOS (positive class)
# XGBoost convention: scale_pos_weight = sum(negative) / sum(positive).
# Class 0 is "negative" (LOS+NLOS) and class 1 is "positive" (NLOS+NLOS).
# This upweights the minority positive class when NLOS+NLOS is underrepresented.
scale_pos_weight = float(n_class0 / max(n_class1, 1))

default_params = {
    "n_estimators": N_ESTIMATORS,
    "max_depth": MAX_DEPTH,
    "learning_rate": LEARNING_RATE,
    "subsample": SUBSAMPLE,
    "colsample_bytree": COLSAMPLE_BYTREE,
    "min_child_weight": MIN_CHILD_WEIGHT,
    "gamma": GAMMA,
    "reg_alpha": REG_ALPHA,
    "reg_lambda": REG_LAMBDA,
}

best_params = default_params.copy()
best_cv_f1 = None
tuning_time = 0.0
tuning_used = False

print("Base configuration:")
for key, value in default_params.items():
    print(f"  {key:16s}: {value}")
print(f"  scale_pos_weight: {scale_pos_weight:.4f}")
print()

if RUN_TUNING:
    optuna_module = None
    try:
        optuna_module = importlib.import_module("optuna")
    except ImportError:
        optuna_module = None

    if optuna_module is None:
        print("Optuna is not installed. Using base parameters instead.")
    else:
        tuning_used = True
        cv_strategy = StratifiedKFold(
            n_splits=PAIR_CV_SPLITS,
            shuffle=True,
            random_state=RANDOM_SEED,
        )
        sampler = optuna_module.samplers.TPESampler(seed=RANDOM_SEED)
        study = optuna_module.create_study(direction="maximize", sampler=sampler)

        print(
            f"Running Optuna tuning ({PAIR_BAYES_TRIALS} trials, {PAIR_CV_SPLITS}-fold CV, objective=F1)..."
        )

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            }

            model = XGBClassifier(
                **params,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method=XGB_TREE_METHOD,
                device=XGB_DEVICE,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_SEED,
                n_jobs=1,
                verbosity=0,
            )

            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv_strategy,
                scoring="f1",
                n_jobs=1 if USE_GPU else -1,
            )
            return float(np.mean(scores))

        t0 = time.time()
        study.optimize(objective, n_trials=PAIR_BAYES_TRIALS, n_jobs=1)
        tuning_time = time.time() - t0

        best_params = study.best_params
        best_cv_f1 = float(study.best_value)

        print(f"Tuning complete in {tuning_time:.2f}s")
        print(f"Best CV F1: {best_cv_f1:.4f}")
        print("Best parameters:")
        for key, value in best_params.items():
            print(f"  {key:16s}: {value}")
        print()
else:
    print("Tuning disabled (RUN_TUNING=0). Using base parameters.")
    print()

xgb_model = XGBClassifier(
    **best_params,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method=XGB_TREE_METHOD,
    device=XGB_DEVICE,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbosity=1,
)

t0 = time.time()
xgb_model.fit(X_train, y_train)
train_time = time.time() - t0

print(f"XGBoost training complete in {train_time:.2f}s")
print()


# ==============================================================================
# STEP 3: PREDICT + EVALUATE
# ==============================================================================
print("Step 3: Evaluating XGBoost pair classifier...")
print("-" * 80)

y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_proba_xgb >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_xgb)
precision = precision_score(y_test, y_pred_xgb)
recall = recall_score(y_test, y_pred_xgb)
f1 = f1_score(y_test, y_pred_xgb)
auc = roc_auc_score(y_test, y_proba_xgb)

print(f"  Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"  ROC-AUC   : {auc:.4f}")
print()
print(classification_report(y_test, y_pred_xgb, target_names=CLASS_NAMES))

# --------------------------------------------------------------------------
# Per-path classification output (required by project brief)
# The brief asks for LOS/NLOS classification of BOTH dominant paths.
# Path 1: determined by the model (LOS when pair_label=0, NLOS when pair_label=1)
# Path 2: ALWAYS NLOS per brief ("next path will be NLOS regardless")
# --------------------------------------------------------------------------
n_test = len(y_test)
path1_los = int((y_pred_xgb == 0).sum())  # predicted LOS+NLOS → path 1 = LOS
path1_nlos = int((y_pred_xgb == 1).sum())  # predicted NLOS+NLOS → path 1 = NLOS

print("=" * 60)
print("PER-PATH CLASSIFICATION SUMMARY")
print("=" * 60)
print(f"  Path 1 — Predicted LOS  : {path1_los:,} ({path1_los / n_test * 100:.1f}%)")
print(f"  Path 1 — Predicted NLOS : {path1_nlos:,} ({path1_nlos / n_test * 100:.1f}%)")
print(f"  Path 2 — Always NLOS    : {n_test:,} (100.0%)  [deterministic per brief]")
print()
print("  Rule applied:")
print("    pair_label=0 (LOS+NLOS)  → Path1=LOS,  Path2=NLOS")
print("    pair_label=1 (NLOS+NLOS) → Path1=NLOS, Path2=NLOS")
print("=" * 60)
print()


# ==============================================================================
# STEP 4: CONFUSION MATRIX
# ==============================================================================
print("Step 4: Plotting confusion matrix...")

cm = confusion_matrix(y_test, y_pred_xgb)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax,
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    linewidths=0.5,
)
ax.set_title("Pair-Level Confusion Matrix - XGBoost", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted Pair", fontsize=11)
ax.set_ylabel("Actual Pair", fontsize=11)

total = cm.sum()
for i in range(2):
    for j in range(2):
        ax.text(
            j + 0.5,
            i + 0.75,
            f"({cm[i, j] / max(total, 1) * 100:.1f}%)",
            ha="center",
            va="center",
            fontsize=9,
            color="red",
        )

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pair_confusion_matrices.png", dpi=300, bbox_inches="tight")
print("Saved: pair_confusion_matrices.png")
plt.close()


# ==============================================================================
# STEP 5: ROC CURVE
# ==============================================================================
print("Step 5: Plotting ROC curve...")

fpr, tpr, _ = roc_curve(y_test, y_proba_xgb)

fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(
    [0, 1],
    [0, 1],
    color="gray",
    linestyle="--",
    linewidth=1.5,
    label="Random Classifier (AUC = 0.500)",
)
ax.plot(fpr, tpr, color="#e67e22", linewidth=2.5, label=f"XGBoost (AUC = {auc:.4f})")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("Pair-Level ROC Curve - XGBoost", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pair_roc_curve.png", dpi=300, bbox_inches="tight")
print("Saved: pair_roc_curve.png")
plt.close()


# ==============================================================================
# STEP 6: FEATURE IMPORTANCE (XGBOOST)
# ==============================================================================
print("Step 6: Plotting feature importance...")

xgb_imp = (
    pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": xgb_model.feature_importances_,
        }
    )
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

xgb_imp.to_csv(OUTPUT_DIR / "pair_feature_importance_xgb.csv", index=False)
print("Saved: pair_feature_importance_xgb.csv")

top20 = xgb_imp.head(20).copy()


def feature_color(name: str) -> str:
    if name.startswith("PEAK2"):
        return "#e74c3c"
    if name.startswith("CIR"):
        return "#3498db"
    return "#95a5a6"


colors = [feature_color(name) for name in top20["Feature"]]

fig, ax = plt.subplots(figsize=(12, 8))
y_pos = np.arange(len(top20))
ax.barh(y_pos, top20["Importance"], color=colors, alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(top20["Feature"], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Feature Importance (Gain)", fontsize=12)
ax.set_title(
    "Pair Classifier - Top 20 Feature Importance (XGBoost)",
    fontsize=12,
    fontweight="bold",
)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "pair_feature_importance_xgb.png", dpi=300, bbox_inches="tight"
)
print("Saved: pair_feature_importance_xgb.png")
plt.close()


# Category contribution chart
feature_arr = xgb_imp["Feature"].to_numpy(dtype=str)
importance_arr = xgb_imp["Importance"].to_numpy(dtype=float)

peak2_imp = float(importance_arr[np.char.startswith(feature_arr, "PEAK2")].sum())
cir_imp = float(importance_arr[np.char.startswith(feature_arr, "CIR")].sum())
core_imp = float(max(0.0, 1.0 - peak2_imp - cir_imp))

print()
print("  Feature importance breakdown (XGBoost):")
print(f"    2nd-path features (PEAK2_*) : {peak2_imp * 100:.1f}%")
print(f"    CIR waveform features       : {cir_imp * 100:.1f}%")
print(f"    Core physical features      : {core_imp * 100:.1f}%")

fig, ax = plt.subplots(figsize=(7, 5))
categories = ["2nd-path\n(PEAK2_*)", "CIR\nwaveform", "Core\nphysical"]
values = [peak2_imp, cir_imp, core_imp]
bar_colors = ["#e74c3c", "#3498db", "#95a5a6"]

bars = ax.bar(categories, values, color=bar_colors, alpha=0.85)
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f"{val * 100:.1f}%",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

ax.set_ylabel("Total Importance", fontsize=12)
ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1.0)
ax.set_title(
    "Pair Classifier - Feature Importance by Category (XGBoost)",
    fontsize=12,
    fontweight="bold",
)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "pair_importance_by_category.png", dpi=300, bbox_inches="tight"
)
print("Saved: pair_importance_by_category.png")
plt.close()


# ==============================================================================
# STEP 7: ERROR ANALYSIS
# ==============================================================================
print()
print("Step 7: Running error analysis...")

error_mask = y_pred_xgb != y_test
correct_mask = ~error_mask

n_errors = int(error_mask.sum())
n_correct = int(correct_mask.sum())
n_fp = int(((y_test == 0) & (y_pred_xgb == 1)).sum())
n_fn = int(((y_test == 1) & (y_pred_xgb == 0)).sum())

error_probs = y_proba_xgb[error_mask]
boundary_low = min(ERROR_BOUNDARY_LOW, ERROR_BOUNDARY_HIGH)
boundary_high = max(ERROR_BOUNDARY_LOW, ERROR_BOUNDARY_HIGH)

near_boundary = int(
    ((error_probs >= boundary_low) & (error_probs <= boundary_high)).sum()
)
confident_errors = int(
    ((error_probs <= CONFIDENT_ERROR_LOW) | (error_probs >= CONFIDENT_ERROR_HIGH)).sum()
)

feat_diffs = []
if n_errors > 0 and n_correct > 0:
    mean_correct = X_test[correct_mask].mean(axis=0)
    mean_error = X_test[error_mask].mean(axis=0)
    for idx, fname in enumerate(feature_names):
        effect = float(abs(mean_error[idx] - mean_correct[idx]))
        feat_diffs.append(
            (fname, effect, float(mean_correct[idx]), float(mean_error[idx]))
        )
    feat_diffs.sort(key=lambda x: x[1], reverse=True)

with open(OUTPUT_DIR / "pair_error_analysis_xgb.txt", "w") as f:
    f.write("PAIR XGBOOST ERROR ANALYSIS REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total test samples   : {len(y_test):,}\n")
    f.write(
        f"Correctly classified : {n_correct:,} ({n_correct / len(y_test) * 100:.1f}%)\n"
    )
    f.write(
        f"Misclassified        : {n_errors:,} ({n_errors / len(y_test) * 100:.1f}%)\n"
    )
    f.write(f"  False Positives (LOS+NLOS predicted as NLOS+NLOS) : {n_fp:,}\n")
    f.write(f"  False Negatives (NLOS+NLOS predicted as LOS+NLOS) : {n_fn:,}\n\n")
    f.write("CONFIDENCE ANALYSIS:\n")
    f.write(
        f"  Errors near boundary ({boundary_low:.2f}-{boundary_high:.2f}): "
        f"{near_boundary:,} ({near_boundary / max(n_errors, 1) * 100:.1f}%)\n"
    )
    f.write(
        f"  Confident errors (<= {CONFIDENT_ERROR_LOW:.2f} or >= {CONFIDENT_ERROR_HIGH:.2f}): "
        f"{confident_errors:,} ({confident_errors / max(n_errors, 1) * 100:.1f}%)\n"
    )
    if n_errors > 0:
        f.write("\n  Error probability stats:\n")
        f.write(f"    Mean  : {np.mean(error_probs):.4f}\n")
        f.write(f"    Median: {np.median(error_probs):.4f}\n")
        f.write(f"    Std   : {np.std(error_probs):.4f}\n\n")

    f.write("TOP 10 FEATURES DISTINGUISHING ERRORS:\n")
    if feat_diffs:
        for rank, (fname, effect, m_corr, m_err) in enumerate(feat_diffs[:10], 1):
            f.write(
                f"  {rank:2d}. {fname:15s}  effect={effect:.3f}  "
                f"(correct={m_corr:.4f}, error={m_err:.4f})\n"
            )
    else:
        f.write(
            "  Not available (no errors or no correctly classified samples to compare).\n"
        )

print("Saved: pair_error_analysis_xgb.txt")


# ==============================================================================
# STEP 8: SAVE MODEL, PREDICTIONS, METRICS
# ==============================================================================
print()
print("Step 8: Saving outputs...")

np.save(OUTPUT_DIR / "y_pred_xgb.npy", y_pred_xgb)
np.save(OUTPUT_DIR / "y_proba_xgb.npy", y_proba_xgb)
np.save(OUTPUT_DIR / "y_test_pair.npy", y_test)

joblib.dump(xgb_model, OUTPUT_DIR / "pair_xgb_model.pkl", compress=3)

metrics_df = pd.DataFrame(
    [
        {
            "Model": "Pair XGBoost",
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": auc,
        }
    ]
)
metrics_df.to_csv(OUTPUT_DIR / "pair_metrics.csv", index=False)

with open(OUTPUT_DIR / "pair_results.txt", "w") as f:
    f.write("PAIR-LEVEL CLASSIFIER RESULTS (XGBOOST ONLY)\n")
    f.write("=" * 60 + "\n\n")
    f.write("Task: LOS+NLOS vs NLOS+NLOS pair classification\n")
    f.write("Model: XGBoost\n")
    f.write("Features: 20 core+second-path + 120 CIR = 140 total\n")
    f.write(f"Training samples: {len(X_train):,}\n")
    f.write(f"Test samples    : {len(X_test):,}\n\n")

    f.write("PAIR_LABEL definition:\n")
    f.write("  0 = LOS+NLOS  (Path 1 LOS exists - trustworthy path available)\n")
    f.write("  1 = NLOS+NLOS (Both paths obstructed - apply bias correction)\n\n")

    f.write("MODEL CONFIGURATION:\n")
    for key, value in best_params.items():
        f.write(f"  {key:16s}: {value}\n")
    f.write(f"  scale_pos_weight: {scale_pos_weight:.4f}\n")
    f.write(f"  random_state    : {RANDOM_SEED}\n")
    f.write(f"  run_tuning      : {RUN_TUNING}\n")
    f.write(f"  tuning_used     : {tuning_used}\n")
    f.write(f"  pair_bayes_trials: {PAIR_BAYES_TRIALS}\n")
    f.write(f"  pair_cv_splits  : {PAIR_CV_SPLITS}\n")
    if best_cv_f1 is not None:
        f.write(f"  best_cv_f1      : {best_cv_f1:.4f}\n")
        f.write(f"  tuning_time_s   : {tuning_time:.2f}\n")
    f.write(f"  training_time_s : {train_time:.2f}\n\n")

    f.write("PERFORMANCE METRICS:\n")
    f.write(f"  Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
    f.write(f"  Precision : {precision:.4f}\n")
    f.write(f"  Recall    : {recall:.4f}\n")
    f.write(f"  F1-Score  : {f1:.4f}\n")
    f.write(f"  ROC-AUC   : {auc:.4f}\n\n")

    f.write("CONFUSION MATRIX:\n")
    f.write(f"  True LOS+NLOS  predicted as LOS+NLOS  : {cm[0, 0]:,}\n")
    f.write(f"  True LOS+NLOS  predicted as NLOS+NLOS : {cm[0, 1]:,}\n")
    f.write(f"  True NLOS+NLOS predicted as LOS+NLOS  : {cm[1, 0]:,}\n")
    f.write(f"  True NLOS+NLOS predicted as NLOS+NLOS : {cm[1, 1]:,}\n\n")

    f.write("PER-PATH CLASSIFICATION (project brief requirement):\n")
    f.write("  Path 1 (predicted by model):\n")
    f.write(f"    Predicted LOS  : {path1_los:,} ({path1_los / n_test * 100:.1f}%)\n")
    f.write(f"    Predicted NLOS : {path1_nlos:,} ({path1_nlos / n_test * 100:.1f}%)\n")
    f.write("  Path 2 (deterministic per brief):\n")
    f.write(f"    Always NLOS    : {n_test:,} (100.0%)\n")
    f.write("  Rule: pair_label=0 → Path1=LOS, Path2=NLOS\n")
    f.write("        pair_label=1 → Path1=NLOS, Path2=NLOS\n\n")

    f.write("FEATURE IMPORTANCE BREAKDOWN:\n")
    f.write(f"  2nd-path features (PEAK2_*) : {peak2_imp * 100:.1f}%\n")
    f.write(f"  CIR waveform features       : {cir_imp * 100:.1f}%\n")
    f.write(f"  Core physical features      : {core_imp * 100:.1f}%\n\n")

    f.write("TOP 10 FEATURES (XGBOOST GAIN):\n")
    for rank, (_, row) in enumerate(xgb_imp.head(10).iterrows(), start=1):
        f.write(f"  {rank:2d}. {row['Feature']}: {row['Importance']:.6f}\n")

print("Saved: y_pred_xgb.npy / y_proba_xgb.npy / y_test_pair.npy")
print("Saved: pair_xgb_model.pkl")
print("Saved: pair_metrics.csv")
print("Saved: pair_results.txt")
print("Saved: pair_error_analysis_xgb.txt")


# ==============================================================================
# SUMMARY
# ==============================================================================
print()
print("=" * 80)
print("PAIR CLASSIFIER COMPLETE - XGBOOST ONLY")
print("=" * 80)
print("RESULTS SUMMARY:")
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"  ROC-AUC   : {auc:.4f}")
print()
print("FILES SAVED TO: models/pair_classifier/")
for fn in [
    "pair_xgb_model.pkl",
    "y_pred_xgb.npy",
    "y_proba_xgb.npy",
    "y_test_pair.npy",
    "pair_confusion_matrices.png",
    "pair_roc_curve.png",
    "pair_feature_importance_xgb.csv",
    "pair_feature_importance_xgb.png",
    "pair_importance_by_category.png",
    "pair_metrics.csv",
    "pair_results.txt",
    "pair_error_analysis_xgb.txt",
]:
    print(f"  - {fn}")
print("=" * 80)

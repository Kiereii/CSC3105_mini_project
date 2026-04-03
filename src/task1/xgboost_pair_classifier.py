"""
Task 1: Pair-Level LOS/NLOS Classifier using XGBoost
=====================================================
Classifies whether a UWB measurement contains a LOS+NLOS pair (label 0)
or an NLOS+NLOS pair (label 1) using XGBoost with stratified cross-validation.

Features: 20 core/second-path features + 120 CIR waveform samples.
"""

import os
import time
import warnings
import importlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]


DATA_DIR = PROJECT_ROOT / "outputs" / "preprocessed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pair_classifier"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = OUTPUT_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
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

print("=" * 80)
print("PAIR-LEVEL CLASSIFIER - XGBOOST ONLY")
print("=" * 80)
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
        if stripped.startswith("Aligned CIR Features"):
            # Aligned window: features are relative to FP_IDX, not absolute CIR indices.
            # Extract the count from the header, e.g. "Aligned CIR Features (120):"
            reading_core = False
            try:
                count = int(stripped.split("(")[1].split(")")[0])
            except (IndexError, ValueError):
                count = 120
            for i in range(count):
                names.append(f"CIR_aligned_{i}")
            break
        if stripped.startswith("CIR Features"):
            reading_core = False
            continue
        if stripped.startswith("Range: CIR"):
            # Expected format: "Range: CIR730 to CIR849"
            try:
                parts = stripped.replace("Range:", "").strip().split()
                cir_start = int(parts[0].replace("CIR", ""))
                cir_end = int(parts[2].replace("CIR", "")) + 1
            except (IndexError, ValueError):
                cir_start, cir_end = 730, 850
            for i in range(cir_start, cir_end):
                names.append(f"CIR{i}")
            break

    # If CIR block was never found, append a fallback range
    if cir_start is None and not any(n.startswith("CIR") for n in names):
        for i in range(120):
            names.append(f"CIR_aligned_{i}")

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
                tree_method="hist",
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
                n_jobs=-1,
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
    tree_method="hist",
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
# STEP 4: SAVE CONFUSION MATRIX DATA
# ==============================================================================
print("Step 4: Saving confusion matrix data...")

cm = confusion_matrix(y_test, y_pred_xgb)
np.save(ARTIFACT_DIR / "pair_confusion_matrix.npy", cm)
print("Saved: artifacts/pair_confusion_matrix.npy")


# ==============================================================================
# STEP 5: SAVE ROC CURVE DATA
# ==============================================================================
print("Step 5: Saving ROC curve data...")

fpr, tpr, _ = roc_curve(y_test, y_proba_xgb)
np.savez(ARTIFACT_DIR / "pair_roc_curve_data.npz", fpr=fpr, tpr=tpr)
print("Saved: artifacts/pair_roc_curve_data.npz")


# ==============================================================================
# STEP 6: SAVE FEATURE IMPORTANCE DATA
# ==============================================================================
print("Step 6: Saving feature importance data...")

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

pd.DataFrame(
    [
        {"Category": "2nd-path (PEAK2_*)", "Importance": peak2_imp},
        {"Category": "CIR waveform", "Importance": cir_imp},
        {"Category": "Core physical", "Importance": core_imp},
    ]
).to_csv(OUTPUT_DIR / "pair_feature_importance_by_category_xgb.csv", index=False)
print("Saved: pair_feature_importance_by_category_xgb.csv")


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
boundary_low = 0.4
boundary_high = 0.6

near_boundary = int(
    ((error_probs >= boundary_low) & (error_probs <= boundary_high)).sum()
)
confident_errors = int(
    ((error_probs <= 0.2) | (error_probs >= 0.8)).sum()
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

error_summary_df = pd.DataFrame(
    [
        {
            "total_test_samples": len(y_test),
            "correctly_classified": n_correct,
            "misclassified": n_errors,
            "false_positives": n_fp,
            "false_negatives": n_fn,
            "boundary_low": boundary_low,
            "boundary_high": boundary_high,
            "near_boundary_errors": near_boundary,
            "confident_errors": confident_errors,
            "error_prob_mean": float(np.mean(error_probs)) if n_errors > 0 else np.nan,
            "error_prob_median": float(np.median(error_probs))
            if n_errors > 0
            else np.nan,
            "error_prob_std": float(np.std(error_probs)) if n_errors > 0 else np.nan,
        }
    ]
)
error_summary_df.to_csv(ARTIFACT_DIR / "pair_error_summary.csv", index=False)
print("Saved: artifacts/pair_error_summary.csv")

pd.DataFrame(
    feat_diffs,
    columns=["feature", "effect", "correct_mean", "error_mean"],
).to_csv(ARTIFACT_DIR / "pair_top_error_features.csv", index=False)
print("Saved: artifacts/pair_top_error_features.csv")

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
        f"  Confident errors (<= 0.20 or >= 0.80): "
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

np.save(ARTIFACT_DIR / "y_pred_xgb.npy", y_pred_xgb)
np.save(ARTIFACT_DIR / "y_proba_xgb.npy", y_proba_xgb)
np.save(ARTIFACT_DIR / "y_test_pair.npy", y_test)

joblib.dump(xgb_model, ARTIFACT_DIR / "pair_xgb_model.pkl", compress=3)

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
    f.write(f"Features: {X_train.shape[1]} total (core+second-path + aligned CIR)\n")
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

print(
    "Saved: artifacts/y_pred_xgb.npy / artifacts/y_proba_xgb.npy / artifacts/y_test_pair.npy"
)
print("Saved: artifacts/pair_xgb_model.pkl")
print("Saved: pair_metrics.csv")
print("Saved: pair_results.txt")
print("Saved: pair_error_analysis_xgb.txt")
print("Saved: pair_feature_importance_by_category_xgb.csv")


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
print(f"FILES SAVED TO: {OUTPUT_DIR}/")
for fn in [
    "pair_feature_importance_xgb.csv",
    "pair_feature_importance_by_category_xgb.csv",
    "pair_metrics.csv",
    "pair_results.txt",
    "pair_error_analysis_xgb.txt",
]:
    print(f"  - {fn}")
print(f"ARTIFACTS SAVED TO: {ARTIFACT_DIR}/")
for fn in [
    "pair_xgb_model.pkl",
    "pair_confusion_matrix.npy",
    "pair_roc_curve_data.npz",
    "pair_error_summary.csv",
    "pair_top_error_features.csv",
    "y_pred_xgb.npy",
    "y_proba_xgb.npy",
    "y_test_pair.npy",
]:
    print(f"  - {fn}")
print("Use notebooks/pair_classifier_analysis.ipynb to regenerate plots.")
print("=" * 80)

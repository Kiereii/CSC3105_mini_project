"""
UWB LOS/NLOS Classification - XGBoost Implementation

Why XGBoost for this problem:
1. Boosting vs Bagging: Learns sequentially from previous errors, unlike
   Random Forest which builds trees in parallel independently. This makes
   it directly comparable and complementary to RF.
2. Widely used in UWB NLOS literature - shown to outperform RF and SVM
   in multiclass NLOS identification tasks. [XGBoost-based Multiclass
   NLOS Channels Identification in UWB, TechScience 2025]
3. No scaling needed - works on unscaled data like Random Forest
4. Native feature importance - can compare directly with RF rankings
5. Handles the 136-feature space efficiently with built-in regularisation
   (L1/L2) to prevent overfitting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from pathlib import Path
import joblib
import time
import os
import warnings
import importlib

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# CONFIGURATION
DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "xgboost"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
RUN_TUNING = os.getenv("RUN_TUNING", "1") == "1"
TUNING_METHOD = os.getenv("TUNING_METHOD", "random").strip().lower()
RANDOM_SEARCH_ITER = int(os.getenv("RANDOM_SEARCH_ITER", "80"))
BAYES_TRIALS = int(os.getenv("BAYES_TRIALS", "80"))
USE_GPU = os.getenv("USE_GPU", "0") == "1"
XGB_DEVICE = "cuda" if USE_GPU else "cpu"
XGB_TREE_METHOD = "hist"

if TUNING_METHOD not in {"random", "bayes"}:
    print(
        f"Warning: Unknown TUNING_METHOD='{TUNING_METHOD}'. Falling back to 'random'."
    )
    TUNING_METHOD = "random"

print("XGBOOST")
print(f"Backend: {'GPU (CUDA)' if USE_GPU else 'CPU'}")
print()

X_train = np.load(DATA_DIR / "X_train_unscaled.npy")
X_val = np.load(DATA_DIR / "X_val_unscaled.npy")
X_test = np.load(DATA_DIR / "X_test_unscaled.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
y_val = np.load(DATA_DIR / "y_val.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

with open(DATA_DIR / "feature_names.txt", "r") as f:
    lines = f.readlines()

feature_names = []
reading_section = None
for line in lines:
    if "Core Features" in line:
        reading_section = "core"
        continue
    if "CIR Features" in line:
        for i in range(730, 850):
            feature_names.append(f"CIR{i}")
        reading_section = "cir"
        continue
    if "Engineered Features" in line:
        reading_section = "engineered"
        continue
    if reading_section in ("core", "engineered") and line.strip().startswith(
        tuple("0123456789")
    ):
        feat_name = line.split(".")[-1].strip()
        feature_names.append(feat_name)

print(f"Training set   : {X_train.shape}")
print(f"Validation set : {X_val.shape}")
print(f"Test set       : {X_test.shape}")
print(f"Features       : {len(feature_names)}")
print(f"  LOS  in train : {np.sum(y_train == 0):,}")
print(f"  NLOS in train : {np.sum(y_train == 1):,}")
print()

# HYPERPARAMETER TUNING (RandomizedSearchCV)
# searches across key XGBoost hyperparameters to find best config
print("Step 2: Hyperparameter Tuning...")
print("-" * 80)

# Compute scale_pos_weight to handle any class imbalance
# (ratio of NLOS to LOS - if perfectly balanced this equals 1.0)
n_los = np.sum(y_train == 0)
n_nlos = np.sum(y_train == 1)
scale_pos_weight = n_los / n_nlos
print(
    f"  scale_pos_weight = {scale_pos_weight:.4f}  "
    f"({'balanced' if abs(scale_pos_weight - 1.0) < 0.01 else 'adjusted for imbalance'})"
)
print()

if RUN_TUNING:
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

    optuna_module = None
    if TUNING_METHOD == "bayes":
        try:
            optuna_module = importlib.import_module("optuna")
        except ImportError:
            optuna_module = None

    if TUNING_METHOD == "bayes" and optuna_module is None:
        print("Optuna is not installed. Falling back to RandomizedSearchCV.")
        TUNING_METHOD = "random"

    if TUNING_METHOD == "random":
        print(
            f"Running RandomizedSearchCV ({RANDOM_SEARCH_ITER} iterations, {cv_strategy.n_splits}-fold CV)..."
        )
        print()

        param_grid = {
            "n_estimators": [200, 400, 600, 800, 1000],
            "max_depth": [3, 4, 5, 6, 8, 10],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
            "min_child_weight": [1, 3, 5, 7, 10],
            "gamma": [0.0, 0.1, 0.3, 0.5, 1.0],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "reg_alpha": [0.0, 0.01, 0.1, 0.5, 1.0],
            "reg_lambda": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
        }

        base_xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method=XGB_TREE_METHOD,
            device=XGB_DEVICE,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,
        )

        search = RandomizedSearchCV(
            estimator=base_xgb,
            param_distributions=param_grid,
            n_iter=RANDOM_SEARCH_ITER,
            cv=cv_strategy,
            scoring="f1",  # Optimise for F1 (balance precision & recall)
            random_state=RANDOM_SEED,
            n_jobs=1 if USE_GPU else -1,
            verbose=1,
        )

        t0 = time.time()
        search.fit(X_train, y_train)
        tuning_time = time.time() - t0

        best_params = search.best_params_
        best_cv_f1 = search.best_score_

    else:
        if optuna_module is None:
            raise RuntimeError("Optuna not available for Bayesian optimization")

        print(
            f"Running Bayesian optimization ({BAYES_TRIALS} trials, {cv_strategy.n_splits}-fold CV)..."
        )
        print()

        sampler = optuna_module.samplers.TPESampler(seed=RANDOM_SEED)
        study = optuna_module.create_study(direction="maximize", sampler=sampler)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
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
        study.optimize(objective, n_trials=BAYES_TRIALS, n_jobs=1)
        tuning_time = time.time() - t0

        best_params = study.best_params
        best_cv_f1 = study.best_value

    print(f"\n✓ Tuning completed in {tuning_time:.2f} seconds")
    print(f"✓ Best CV F1 score : {best_cv_f1:.4f}")
    print(f"\n  Best hyperparameters found:")
    for k, v in best_params.items():
        print(f"    {k:20s}: {v}")
    print()

else:
    # Sensible defaults if you want to skip tuning and run fast
    best_params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }
    print("(Skipping tuning - using default parameters)")
    print()

# ==============================================================================
# STEP 3: TRAIN FINAL XGBoost MODEL WITH BEST PARAMS
# ==============================================================================
print("Step 3: Training final XGBoost model with best parameters...")
print("-" * 80)

xgb_model = XGBClassifier(
    **best_params,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method=XGB_TREE_METHOD,
    device=XGB_DEVICE,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    early_stopping_rounds=30,
    verbosity=1,
)

t0 = time.time()
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
training_time = time.time() - t0

print(f"\n✓ Training completed in {training_time:.2f} seconds")
print()

# Tune decision threshold on validation set
print("Step 3.5: Tuning decision threshold on validation set...")
val_pred_proba = xgb_model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.10, 0.91, 0.01)
best_threshold = 0.50
best_f1 = -1.0
best_recall = -1.0

for t in thresholds:
    val_pred_t = (val_pred_proba >= t).astype(int)
    f1_t = f1_score(y_val, val_pred_t)
    recall_t = recall_score(y_val, val_pred_t)
    if (f1_t > best_f1) or (f1_t == best_f1 and recall_t > best_recall):
        best_f1 = f1_t
        best_recall = recall_t
        best_threshold = t

print(f"  Best threshold: {best_threshold:.2f}")
print(f"  Validation F1: {best_f1:.4f}")
print(f"  Validation Recall: {best_recall:.4f}")
print()

print("Step 4: Predictions...")

y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]  # Probability of NLOS (class 1)
y_pred = (y_pred_proba >= best_threshold).astype(int)

print("✓ Predictions completed")
print()

# ==============================================================================
# STEP 5: EVALUATE MODEL PERFORMANCE
# ==============================================================================
print("Step 5: Evaluating model performance...")
print("=" * 80)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n PERFORMANCE METRICS:")
print("-" * 40)
print(f"Accuracy  : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")
print()

print(" DETAILED CLASSIFICATION REPORT:")
print("-" * 40)
target_names = ["LOS (0)", "NLOS (1)"]
print(classification_report(y_test, y_pred, target_names=target_names))

# ==============================================================================
# STEP 6: CONFUSION MATRIX + ROC CURVE
# ==============================================================================
print("Step 6: Creating confusion matrix and ROC curve...")

cm = confusion_matrix(y_test, y_pred)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax1,
    xticklabels=["LOS", "NLOS"],
    yticklabels=["LOS", "NLOS"],
)
ax1.set_title("Confusion Matrix – XGBoost", fontsize=14, fontweight="bold")
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
total = cm.sum()
for i in range(2):
    for j in range(2):
        ax1.text(
            j + 0.5,
            i + 0.7,
            f"({cm[i, j] / total * 100:.1f}%)",
            ha="center",
            va="center",
            fontsize=10,
            color="red",
        )

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ax2.plot(fpr, tpr, color="#f39c12", linewidth=2, label=f"ROC Curve (AUC = {auc:.4f})")
ax2.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier")
ax2.set_xlabel("False Positive Rate", fontsize=12)
ax2.set_ylabel("True Positive Rate", fontsize=12)
ax2.set_title("ROC Curve – XGBoost", fontsize=14, fontweight="bold")
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "confusion_matrix_and_roc_xgb.png", dpi=300, bbox_inches="tight"
)
print("✓ Saved: confusion_matrix_and_roc_xgb.png")
plt.close()

# ==============================================================================
# STEP 6.5: ERROR ANALYSIS
# ==============================================================================
print("\nStep 6.5: Error Analysis...")
print("=" * 80)

misclassified_mask = y_pred != y_test
correct_mask = ~misclassified_mask
n_errors = misclassified_mask.sum()
n_correct = correct_mask.sum()

fp_mask = (y_pred == 1) & (y_test == 0)  # Predicted NLOS, actually LOS
fn_mask = (y_pred == 0) & (y_test == 1)  # Predicted LOS, actually NLOS
n_fp = fp_mask.sum()
n_fn = fn_mask.sum()

print(f"\n  Total test samples   : {len(y_test):,}")
print(f"  Correctly classified : {n_correct:,} ({n_correct / len(y_test) * 100:.1f}%)")
print(f"  Misclassified        : {n_errors:,} ({n_errors / len(y_test) * 100:.1f}%)")
print(f"    - False Positives (LOS→NLOS) : {n_fp:,}")
print(f"    - False Negatives (NLOS→LOS) : {n_fn:,}")
print()

# Confidence analysis: how confident was the model when it was wrong?
error_probs = y_pred_proba[misclassified_mask]
correct_probs = y_pred_proba[correct_mask]

# Errors near decision boundary (probability between 0.3 and 0.7)
boundary_low, boundary_high = 0.3, 0.7
near_boundary = np.sum((error_probs >= boundary_low) & (error_probs <= boundary_high))
confident_errors = n_errors - near_boundary

print(f"  CONFIDENCE ANALYSIS:")
print(f"  {'-' * 40}")
print(
    f"  Errors near boundary ({boundary_low}-{boundary_high}): "
    f"{near_boundary:,} / {n_errors:,} ({near_boundary / max(n_errors, 1) * 100:.1f}%)"
)
print(
    f"  Confident errors (outside boundary)  : "
    f"{confident_errors:,} / {n_errors:,} ({confident_errors / max(n_errors, 1) * 100:.1f}%)"
)

if n_errors > 0:
    print(f"\n  Error probability stats:")
    print(f"    Mean  : {np.mean(error_probs):.4f}")
    print(f"    Median: {np.median(error_probs):.4f}")
    print(f"    Std   : {np.std(error_probs):.4f}")
    print(f"    Min   : {np.min(error_probs):.4f}")
    print(f"    Max   : {np.max(error_probs):.4f}")
print()

# Feature analysis: which features differ most between correct and misclassified?
print(f"  TOP FEATURES DISTINGUISHING ERRORS FROM CORRECT:")
print(f"  {'-' * 50}")

feat_diffs = []
for i, fname in enumerate(feature_names):
    mean_correct = np.mean(X_test[correct_mask, i])
    mean_error = np.mean(X_test[misclassified_mask, i])
    std_correct = np.std(X_test[correct_mask, i])
    # Normalised difference (effect size) - avoids division by zero
    if std_correct > 1e-10:
        effect = abs(mean_error - mean_correct) / std_correct
    else:
        effect = 0.0
    feat_diffs.append((fname, effect, mean_correct, mean_error))

feat_diffs.sort(key=lambda x: x[1], reverse=True)

for rank, (fname, effect, m_corr, m_err) in enumerate(feat_diffs[:10], 1):
    print(
        f"    {rank:2d}. {fname:15s}  effect={effect:.3f}  "
        f"(correct_mean={m_corr:.4f}, error_mean={m_err:.4f})"
    )
print()

# Visualisation: 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) Probability distribution: correct vs misclassified
axes[0, 0].hist(
    correct_probs, bins=50, alpha=0.6, color="#2ecc71", label="Correct", density=True
)
axes[0, 0].hist(
    error_probs,
    bins=50,
    alpha=0.6,
    color="#e74c3c",
    label="Misclassified",
    density=True,
)
axes[0, 0].axvline(
    x=best_threshold,
    color="black",
    linestyle="--",
    label=f"Threshold ({best_threshold:.2f})",
)
axes[0, 0].set_xlabel("Predicted Probability (NLOS)")
axes[0, 0].set_ylabel("Density")
axes[0, 0].set_title("Prediction Confidence: Correct vs Errors", fontweight="bold")
axes[0, 0].legend()

# (2) FP vs FN probability distributions
if n_fp > 0:
    axes[0, 1].hist(
        y_pred_proba[fp_mask],
        bins=30,
        alpha=0.6,
        color="#3498db",
        label=f"FP – LOS→NLOS ({n_fp})",
        density=True,
    )
if n_fn > 0:
    axes[0, 1].hist(
        y_pred_proba[fn_mask],
        bins=30,
        alpha=0.6,
        color="#e67e22",
        label=f"FN – NLOS→LOS ({n_fn})",
        density=True,
    )
axes[0, 1].axvline(x=best_threshold, color="black", linestyle="--", label=f"Threshold")
axes[0, 1].set_xlabel("Predicted Probability (NLOS)")
axes[0, 1].set_ylabel("Density")
axes[0, 1].set_title("False Positives vs False Negatives", fontweight="bold")
axes[0, 1].legend()

# (3) Top 10 features with largest effect size (error vs correct)
top_feats = feat_diffs[:10]
feat_labels = [f[0] for f in top_feats][::-1]
feat_effects = [f[1] for f in top_feats][::-1]
axes[1, 0].barh(feat_labels, feat_effects, color="#9b59b6", alpha=0.8)
axes[1, 0].set_xlabel("Effect Size (|mean_diff| / std)")
axes[1, 0].set_title("Features Most Different in Errors", fontweight="bold")
axes[1, 0].grid(axis="x", alpha=0.3)

# (4) Error rate by confidence bin
prob_bins = np.linspace(0, 1, 11)
bin_labels = []
bin_error_rates = []
for b in range(len(prob_bins) - 1):
    in_bin = (y_pred_proba >= prob_bins[b]) & (y_pred_proba < prob_bins[b + 1])
    n_in_bin = in_bin.sum()
    if n_in_bin > 0:
        err_rate = (y_pred[in_bin] != y_test[in_bin]).sum() / n_in_bin
        bin_labels.append(f"{prob_bins[b]:.1f}-{prob_bins[b + 1]:.1f}")
        bin_error_rates.append(err_rate)
axes[1, 1].bar(bin_labels, bin_error_rates, color="#f39c12", alpha=0.8)
axes[1, 1].set_xlabel("Predicted Probability Bin")
axes[1, 1].set_ylabel("Error Rate")
axes[1, 1].set_title("Error Rate by Confidence Bin", fontweight="bold")
axes[1, 1].tick_params(axis="x", rotation=45)
axes[1, 1].grid(axis="y", alpha=0.3)

plt.suptitle("XGBoost Error Analysis", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "error_analysis_xgb.png", dpi=300, bbox_inches="tight")
print("✓ Saved: error_analysis_xgb.png")
plt.close()

# Save error analysis report
with open(OUTPUT_DIR / "error_analysis_xgb.txt", "w") as f:
    f.write("XGBOOST ERROR ANALYSIS REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total test samples   : {len(y_test):,}\n")
    f.write(
        f"Correctly classified : {n_correct:,} ({n_correct / len(y_test) * 100:.1f}%)\n"
    )
    f.write(
        f"Misclassified        : {n_errors:,} ({n_errors / len(y_test) * 100:.1f}%)\n"
    )
    f.write(f"  False Positives (LOS predicted as NLOS) : {n_fp:,}\n")
    f.write(f"  False Negatives (NLOS predicted as LOS) : {n_fn:,}\n\n")
    f.write("CONFIDENCE ANALYSIS:\n")
    f.write(
        f"  Errors near boundary ({boundary_low}-{boundary_high}): {near_boundary:,} ({near_boundary / max(n_errors, 1) * 100:.1f}%)\n"
    )
    f.write(
        f"  Confident errors: {confident_errors:,} ({confident_errors / max(n_errors, 1) * 100:.1f}%)\n"
    )
    if n_errors > 0:
        f.write(f"\n  Error probability stats:\n")
        f.write(f"    Mean  : {np.mean(error_probs):.4f}\n")
        f.write(f"    Median: {np.median(error_probs):.4f}\n")
        f.write(f"    Std   : {np.std(error_probs):.4f}\n\n")
    f.write("TOP 10 FEATURES DISTINGUISHING ERRORS:\n")
    for rank, (fname, effect, m_corr, m_err) in enumerate(feat_diffs[:10], 1):
        f.write(
            f"  {rank:2d}. {fname:15s}  effect={effect:.3f}  (correct={m_corr:.4f}, error={m_err:.4f})\n"
        )

print("✓ Saved: error_analysis_xgb.txt")
print()

# STEP 7: FEATURE IMPORTANCE ANALYSIS
# XGBoost provides 3 importance types - we use 'gain' (most informative)
# gain   = average gain of the feature when it is used in trees
# weight = number of times the feature appears in trees
# cover  = average coverage of the feature across all trees
importances = xgb_model.feature_importances_  # 'gain' by default
indices = np.argsort(importances)[::-1]

feature_importance_df = pd.DataFrame(
    {
        "Feature": [feature_names[i] for i in indices],
        "Importance": importances[indices],
    }
)

print("\n TOP 20 MOST IMPORTANT FEATURES (XGBoost):")
print("-" * 50)
for i in range(min(20, len(feature_names))):
    print(f"{i + 1:2d}. {feature_names[indices[i]]:15s}: {importances[indices[i]]:.4f}")
print()

# Top 20 feature bar chart
fig, ax = plt.subplots(figsize=(12, 8))
top_n = 20
top_indices = indices[:top_n]
top_features = [feature_names[i] for i in top_indices]
top_importances = importances[top_indices]

y_pos = np.arange(len(top_features))
ax.barh(y_pos, top_importances, align="center", color="#f39c12")
ax.set_yticks(y_pos)
ax.set_yticklabels(top_features)
ax.invert_yaxis()
ax.set_xlabel("Feature Importance (Gain)", fontsize=12)
ax.set_title(
    f"Top {top_n} Most Important Features (XGBoost)", fontsize=14, fontweight="bold"
)
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance_xgb.png", dpi=300, bbox_inches="tight")
print("✓ Saved: feature_importance_xgb.png")
plt.close()

# ==============================================================================
# STEP 8: FEATURE CATEGORY ANALYSIS (mirrors RF script exactly)
# ==============================================================================
print("Step 8: Analyzing feature categories...")

core_importance = []
cir_importance = []

for i, feat_name in enumerate(feature_names):
    if feat_name.startswith("CIR"):
        cir_importance.append(importances[i])
    else:
        core_importance.append(importances[i])

core_total = np.sum(core_importance)
cir_total = np.sum(cir_importance)

print(f"\n IMPORTANCE BY CATEGORY:")
print("-" * 40)
print(
    f"Core Features ({len(core_importance)}): {core_total:.4f} ({core_total * 100:.1f}%)"
)
print(
    f"CIR  Features ({len(cir_importance)}): {cir_total:.4f}  ({cir_total * 100:.1f}%)"
)
print()

fig, ax = plt.subplots(figsize=(8, 6))
categories = [
    f"Core Features\n({len(core_importance)} features)",
    f"CIR Features\n({len(cir_importance)} features)",
]
values = [core_total, cir_total]
colors = ["#e74c3c", "#f39c12"]

bars = ax.bar(categories, values, color=colors, alpha=0.8)
ax.set_ylabel("Total Importance", fontsize=12)
ax.set_title("Feature Importance by Category – XGBoost", fontsize=14, fontweight="bold")
ax.set_ylim(0, 1)

for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{val:.3f}\n({val * 100:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "importance_by_category_xgb.png", dpi=300, bbox_inches="tight")
print("✓ Saved: importance_by_category_xgb.png")
plt.close()

# ==============================================================================
# STEP 9: THREE-MODEL COMBINED ROC PLOT (RF + LR + SVM + XGBoost)
# ==============================================================================
print("Step 9: Generating combined ROC plot (all models)...")

rf_proba_path = PROJECT_ROOT / "outputs" / "random_forest" / "y_pred_proba.npy"
lr_proba_path = PROJECT_ROOT / "outputs" / "logreg_svm" / "y_proba_lr.npy"
svm_proba_path = PROJECT_ROOT / "outputs" / "logreg_svm" / "y_proba_svm.npy"

fig, ax = plt.subplots(figsize=(9, 7))
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier")

# XGBoost (always available here)
ax.plot(fpr, tpr, color="#f39c12", linewidth=2, label=f"XGBoost (AUC = {auc:.4f})")

# Load other models if they exist
if rf_proba_path.exists():
    y_proba_rf = np.load(rf_proba_path)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
    auc_rf = roc_auc_score(y_test, y_proba_rf)
    ax.plot(
        fpr_rf,
        tpr_rf,
        color="#2ecc71",
        linewidth=2,
        label=f"Random Forest (AUC = {auc_rf:.4f})",
    )

if lr_proba_path.exists():
    y_proba_lr = np.load(lr_proba_path)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
    auc_lr = roc_auc_score(y_test, y_proba_lr)
    ax.plot(
        fpr_lr,
        tpr_lr,
        color="#e74c3c",
        linewidth=2,
        label=f"Logistic Regression (AUC = {auc_lr:.4f})",
    )

if svm_proba_path.exists():
    y_proba_svm = np.load(svm_proba_path)
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)
    auc_svm = roc_auc_score(y_test, y_proba_svm)
    ax.plot(
        fpr_svm,
        tpr_svm,
        color="#9b59b6",
        linewidth=2,
        label=f"SVM – LinearSVC (AUC = {auc_svm:.4f})",
    )

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve Comparison – All Models", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_all_models.png", dpi=300, bbox_inches="tight")
print("✓ Saved: roc_all_models.png")
plt.close()

# ==============================================================================
# STEP 10: SAVE MODEL AND RESULTS
# ==============================================================================
print("\nStep 10: Saving model and results...")
print("=" * 80)

joblib.dump(xgb_model, OUTPUT_DIR / "xgboost_model.pkl")
np.save(OUTPUT_DIR / "y_pred_xgb.npy", y_pred)
np.save(OUTPUT_DIR / "y_pred_proba_xgb.npy", y_pred_proba)
feature_importance_df.to_csv(OUTPUT_DIR / "feature_importance_xgb.csv", index=False)

print("✓ Saved: xgboost_model.pkl")
print("✓ Saved: y_pred_xgb.npy / y_pred_proba_xgb.npy")
print("✓ Saved: feature_importance_xgb.csv")

with open(OUTPUT_DIR / "model_results_xgb.txt", "w") as f:
    f.write("XGBOOST CLASSIFICATION RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Dataset: UWB LOS/NLOS\n")
    f.write(f"Features: {len(feature_names)} (core + CIR 730-849)\n")
    f.write(f"Training samples: {len(X_train):,}\n")
    f.write(f"Validation samples: {len(X_val):,}\n")
    f.write(f"Test samples    : {len(X_test):,}\n\n")
    f.write("MODEL CONFIGURATION:\n")
    for k, v in best_params.items():
        f.write(f"  - {k}: {v}\n")
    f.write(f"  - scale_pos_weight: {scale_pos_weight:.4f}\n")
    f.write(f"  - decision_threshold: {best_threshold:.2f}\n")
    f.write(f"  - random_state    : {RANDOM_SEED}\n\n")
    f.write("VALIDATION THRESHOLD TUNING:\n")
    f.write(f"  - Best threshold: {best_threshold:.2f}\n")
    f.write(f"  - Validation F1: {best_f1:.4f}\n")
    f.write(f"  - Validation Recall: {best_recall:.4f}\n\n")
    f.write("PERFORMANCE METRICS:\n")
    f.write(f"  - Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
    f.write(f"  - Precision : {precision:.4f}\n")
    f.write(f"  - Recall    : {recall:.4f}\n")
    f.write(f"  - F1-Score  : {f1:.4f}\n")
    f.write(f"  - ROC-AUC   : {auc:.4f}\n\n")
    f.write("FEATURE IMPORTANCE SUMMARY:\n")
    f.write(
        f"  - Core features importance : {core_total:.4f} ({core_total * 100:.1f}%)\n"
    )
    f.write(
        f"  - CIR  features importance : {cir_total:.4f}  ({cir_total * 100:.1f}%)\n\n"
    )
    f.write("CONFUSION MATRIX:\n")
    f.write(f"  True LOS  predicted as LOS  : {cm[0, 0]:,}\n")
    f.write(f"  True LOS  predicted as NLOS : {cm[0, 1]:,}\n")
    f.write(f"  True NLOS predicted as LOS  : {cm[1, 0]:,}\n")
    f.write(f"  True NLOS predicted as NLOS : {cm[1, 1]:,}\n\n")
    f.write("TOP 5 FEATURES:\n")
    for rank in range(5):
        f.write(
            f"  {rank + 1}. {feature_names[indices[rank]]}: "
            f"{importances[indices[rank]]:.4f}\n"
        )

print("✓ Saved: model_results_xgb.txt")

# ==============================================================================
# STEP 11: SUMMARY
# ==============================================================================
print()
print("=" * 80)
print("XGBOOST CLASSIFICATION - COMPLETE!")
print("=" * 80)
print()
print(" KEY RESULTS:")
print(f"   Accuracy  : {accuracy * 100:.2f}%")
print(f"   F1-Score  : {f1:.4f}")
print(f"   ROC-AUC   : {auc:.4f}")
print()
print(" KEY INSIGHTS:")
print(
    f"   Top feature : {feature_names[indices[0]]} "
    f"(importance = {importances[indices[0]]:.4f})"
)
print(f"   Core features contribute : {core_total * 100:.1f}% of total importance")
print(f"   CIR  features contribute : {cir_total * 100:.1f}% of total importance")
print()
print(" Generated files:")
for f in [
    "xgboost_model.pkl",
    "confusion_matrix_and_roc_xgb.png",
    "error_analysis_xgb.png",
    "feature_importance_xgb.png",
    "importance_by_category_xgb.png",
    "roc_all_models.png",
    "model_results_xgb.txt",
    "error_analysis_xgb.txt",
    "feature_importance_xgb.csv",
]:
    print(f"   - {f}")
print()
print("=" * 80)

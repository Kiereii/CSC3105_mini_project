"""
UWB LOS/NLOS Classification - Random Forest Implementation

Why Random Forest for this problem:
1. Bagging approach: Builds trees in parallel independently, each on a
   bootstrapped subset. Unlike XGBoost (sequential boosting), RF reduces
   variance through averaging, making it directly complementary.
2. No scaling needed - works on unscaled data (same as XGBoost)
3. Native feature importance - can compare directly with XGBoost rankings
4. Robust to outliers and handles the 136-feature space well
5. Fewer hyperparameters to tune than gradient boosting methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
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

# CONFIGURATION
RUN_NAME = os.getenv("RUN_NAME", "split_env_70_15_15_seed42")
DATA_DIR = Path("./runs") / RUN_NAME / "preprocessed_data"
OUTPUT_DIR = Path("./runs") / RUN_NAME / "models" / "random_forest"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
RUN_TUNING = os.getenv("RUN_TUNING", "1") == "1"
TUNING_METHOD = os.getenv("TUNING_METHOD", "random").strip().lower()
RANDOM_SEARCH_ITER = int(os.getenv("RANDOM_SEARCH_ITER", "80"))
BAYES_TRIALS = int(os.getenv("BAYES_TRIALS", "80"))

if TUNING_METHOD not in {"random", "bayes"}:
    print(
        f"Warning: Unknown TUNING_METHOD='{TUNING_METHOD}'. Falling back to 'random'."
    )
    TUNING_METHOD = "random"

print("=" * 80)
print("RANDOM FOREST CLASSIFIER - UWB LOS/NLOS PREDICTION")
print("=" * 80)
print()

# =============================================================================
# STEP 1: LOAD PREPROCESSED DATA
# =============================================================================
print("Step 1: Loading preprocessed data...")
print()

X_train = np.load(DATA_DIR / "X_train_unscaled.npy")
X_val = np.load(DATA_DIR / "X_val_unscaled.npy")
X_test = np.load(DATA_DIR / "X_test_unscaled.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
y_val = np.load(DATA_DIR / "y_val.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

# Load feature names
with open(DATA_DIR / "feature_names.txt", "r") as f:
    lines = f.readlines()

feature_names = []
reading_features = False
for line in lines:
    if "Core Features" in line:
        reading_features = True
        continue
    if reading_features and line.strip().startswith(tuple("0123456789")):
        feat_name = line.split(".")[-1].strip()
        feature_names.append(feat_name)
    if "CIR Features" in line:
        for i in range(730, 850):
            feature_names.append(f"CIR{i}")
        break

print(f"Training set   : {X_train.shape}")
print(f"Validation set : {X_val.shape}")
print(f"Test set       : {X_test.shape}")
print(f"Features       : {len(feature_names)}")
print(f"  LOS  in train : {np.sum(y_train == 0):,}")
print(f"  NLOS in train : {np.sum(y_train == 1):,}")
print()

# =============================================================================
# STEP 2: HYPERPARAMETER TUNING
# =============================================================================
print("Step 2: Hyperparameter Tuning...")
print("-" * 80)

# Compute class weight ratio for reference
n_los = np.sum(y_train == 0)
n_nlos = np.sum(y_train == 1)
class_ratio = n_los / n_nlos
print(
    f"  class ratio (LOS/NLOS) = {class_ratio:.4f}  "
    f"({'balanced' if abs(class_ratio - 1.0) < 0.01 else 'imbalanced - will test class_weight options'})"
)
print()

if RUN_TUNING:
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

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
            "n_estimators": [100, 200, 400, 600, 800, 1000],
            "max_depth": [None, 10, 15, 20, 25, 30],
            "min_samples_split": [2, 5, 10, 15, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
            "class_weight": [None, "balanced", "balanced_subsample"],
        }

        base_rf = RandomForestClassifier(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=0,
        )

        search = RandomizedSearchCV(
            estimator=base_rf,
            param_distributions=param_grid,
            n_iter=RANDOM_SEARCH_ITER,
            cv=cv_strategy,
            scoring="f1",
            random_state=RANDOM_SEED,
            n_jobs=-1,
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

        print(f"Running Bayesian optimization ({BAYES_TRIALS} trials, {cv_strategy.n_splits}-fold CV)...")
        print()

        sampler = optuna_module.samplers.TPESampler(seed=RANDOM_SEED)
        study = optuna_module.create_study(direction="maximize", sampler=sampler)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "max_depth": trial.suggest_int("max_depth", 5, 40),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]
                ),
                "class_weight": trial.suggest_categorical(
                    "class_weight", ["None", "balanced", "balanced_subsample"]
                ),
            }

            # Convert "None" string back to None
            if params["class_weight"] == "None":
                params["class_weight"] = None

            model = RandomForestClassifier(
                **params,
                random_state=RANDOM_SEED,
                n_jobs=1,
                verbose=0,
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
        study.optimize(objective, n_trials=BAYES_TRIALS, n_jobs=1)
        tuning_time = time.time() - t0

        best_params = study.best_params.copy()
        best_cv_f1 = study.best_value

        # Convert "None" string back to None for class_weight
        if best_params.get("class_weight") == "None":
            best_params["class_weight"] = None

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
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": None,
    }
    print("(Skipping tuning - using default parameters)")
    print()

# =============================================================================
# STEP 3: TRAIN FINAL RANDOM FOREST MODEL WITH BEST PARAMS
# =============================================================================
print("Step 3: Training final Random Forest model with best parameters...")
print("-" * 80)

rf_model = RandomForestClassifier(
    **best_params,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=1,
)

t0 = time.time()
rf_model.fit(X_train, y_train)
training_time = time.time() - t0

print(f"\n✓ Training completed in {training_time:.2f} seconds")
print(f"✓ Number of trees: {rf_model.n_estimators}")
print()

# =============================================================================
# STEP 3.5: TUNE DECISION THRESHOLD ON VALIDATION SET
# =============================================================================
print("Step 3.5: Tuning decision threshold on validation set...")

val_pred_proba = rf_model.predict_proba(X_val)[:, 1]

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

# =============================================================================
# STEP 4: MAKE PREDICTIONS
# =============================================================================
print("Step 4: Predictions...")

y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # Probability of NLOS (class 1)
y_pred = (y_pred_proba >= best_threshold).astype(int)

print("✓ Predictions completed")
print()

# =============================================================================
# STEP 5: EVALUATE MODEL PERFORMANCE
# =============================================================================
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

# =============================================================================
# STEP 6: CONFUSION MATRIX + ROC CURVE
# =============================================================================
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
ax1.set_title("Confusion Matrix – Random Forest", fontsize=14, fontweight="bold")
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
ax2.plot(fpr, tpr, color="#2ecc71", linewidth=2, label=f"ROC Curve (AUC = {auc:.4f})")
ax2.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier")
ax2.set_xlabel("False Positive Rate", fontsize=12)
ax2.set_ylabel("True Positive Rate", fontsize=12)
ax2.set_title("ROC Curve – Random Forest", fontsize=14, fontweight="bold")
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "confusion_matrix_and_roc.png", dpi=300, bbox_inches="tight"
)
print("✓ Saved: confusion_matrix_and_roc.png")
plt.close()

# =============================================================================
# STEP 6.5: ERROR ANALYSIS
# =============================================================================
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

# Confidence analysis
error_probs = y_pred_proba[misclassified_mask]
correct_probs = y_pred_proba[correct_mask]

boundary_low, boundary_high = 0.3, 0.7
near_boundary = np.sum(
    (error_probs >= boundary_low) & (error_probs <= boundary_high)
)
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
    error_probs, bins=50, alpha=0.6, color="#e74c3c", label="Misclassified", density=True
)
axes[0, 0].axvline(x=best_threshold, color="black", linestyle="--", label=f"Threshold ({best_threshold:.2f})")
axes[0, 0].set_xlabel("Predicted Probability (NLOS)")
axes[0, 0].set_ylabel("Density")
axes[0, 0].set_title("Prediction Confidence: Correct vs Errors", fontweight="bold")
axes[0, 0].legend()

# (2) FP vs FN probability distributions
if n_fp > 0:
    axes[0, 1].hist(
        y_pred_proba[fp_mask], bins=30, alpha=0.6, color="#3498db",
        label=f"FP – LOS→NLOS ({n_fp})", density=True
    )
if n_fn > 0:
    axes[0, 1].hist(
        y_pred_proba[fn_mask], bins=30, alpha=0.6, color="#e67e22",
        label=f"FN – NLOS→LOS ({n_fn})", density=True
    )
axes[0, 1].axvline(x=best_threshold, color="black", linestyle="--", label="Threshold")
axes[0, 1].set_xlabel("Predicted Probability (NLOS)")
axes[0, 1].set_ylabel("Density")
axes[0, 1].set_title("False Positives vs False Negatives", fontweight="bold")
axes[0, 1].legend()

# (3) Top 10 features with largest effect size
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
        bin_labels.append(f"{prob_bins[b]:.1f}-{prob_bins[b+1]:.1f}")
        bin_error_rates.append(err_rate)
axes[1, 1].bar(bin_labels, bin_error_rates, color="#2ecc71", alpha=0.8)
axes[1, 1].set_xlabel("Predicted Probability Bin")
axes[1, 1].set_ylabel("Error Rate")
axes[1, 1].set_title("Error Rate by Confidence Bin", fontweight="bold")
axes[1, 1].tick_params(axis="x", rotation=45)
axes[1, 1].grid(axis="y", alpha=0.3)

plt.suptitle("Random Forest Error Analysis", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "error_analysis_rf.png", dpi=300, bbox_inches="tight")
print("✓ Saved: error_analysis_rf.png")
plt.close()

# Save error analysis report
with open(OUTPUT_DIR / "error_analysis_rf.txt", "w") as f:
    f.write("RANDOM FOREST ERROR ANALYSIS REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total test samples   : {len(y_test):,}\n")
    f.write(f"Correctly classified : {n_correct:,} ({n_correct / len(y_test) * 100:.1f}%)\n")
    f.write(f"Misclassified        : {n_errors:,} ({n_errors / len(y_test) * 100:.1f}%)\n")
    f.write(f"  False Positives (LOS predicted as NLOS) : {n_fp:,}\n")
    f.write(f"  False Negatives (NLOS predicted as LOS) : {n_fn:,}\n\n")
    f.write("CONFIDENCE ANALYSIS:\n")
    f.write(f"  Errors near boundary ({boundary_low}-{boundary_high}): {near_boundary:,} ({near_boundary / max(n_errors, 1) * 100:.1f}%)\n")
    f.write(f"  Confident errors: {confident_errors:,} ({confident_errors / max(n_errors, 1) * 100:.1f}%)\n")
    if n_errors > 0:
        f.write(f"\n  Error probability stats:\n")
        f.write(f"    Mean  : {np.mean(error_probs):.4f}\n")
        f.write(f"    Median: {np.median(error_probs):.4f}\n")
        f.write(f"    Std   : {np.std(error_probs):.4f}\n\n")
    f.write("TOP 10 FEATURES DISTINGUISHING ERRORS:\n")
    for rank, (fname, effect, m_corr, m_err) in enumerate(feat_diffs[:10], 1):
        f.write(f"  {rank:2d}. {fname:15s}  effect={effect:.3f}  (correct={m_corr:.4f}, error={m_err:.4f})\n")

print("✓ Saved: error_analysis_rf.txt")
print()

# =============================================================================
# STEP 7: FEATURE IMPORTANCE ANALYSIS
# =============================================================================
print("\nStep 7: Analyzing feature importance...")
print("=" * 80)

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

feature_importance_df = pd.DataFrame(
    {
        "Feature": [feature_names[i] for i in indices],
        "Importance": importances[indices],
    }
)

print("\n TOP 20 MOST IMPORTANT FEATURES (Random Forest):")
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
ax.barh(y_pos, top_importances, align="center", color="#2ecc71")
ax.set_yticks(y_pos)
ax.set_yticklabels(top_features)
ax.invert_yaxis()
ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
ax.set_title(
    f"Top {top_n} Most Important Features (Random Forest)", fontsize=14, fontweight="bold"
)
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=300, bbox_inches="tight")
print("✓ Saved: feature_importance.png")
plt.close()

# =============================================================================
# STEP 8: FEATURE CATEGORY ANALYSIS
# =============================================================================
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
colors = ["#e74c3c", "#2ecc71"]

bars = ax.bar(categories, values, color=colors, alpha=0.8)
ax.set_ylabel("Total Importance", fontsize=12)
ax.set_title("Feature Importance by Category – Random Forest", fontsize=14, fontweight="bold")
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
plt.savefig(OUTPUT_DIR / "importance_by_category.png", dpi=300, bbox_inches="tight")
print("✓ Saved: importance_by_category.png")
plt.close()

# =============================================================================
# STEP 9: SAVE MODEL AND RESULTS
# =============================================================================
print("\nStep 9: Saving model and results...")
print("=" * 80)

joblib.dump(rf_model, OUTPUT_DIR / "random_forest_model.pkl")
np.save(OUTPUT_DIR / "y_pred.npy", y_pred)
np.save(OUTPUT_DIR / "y_pred_proba.npy", y_pred_proba)
feature_importance_df.to_csv(OUTPUT_DIR / "feature_importance_ranking.csv", index=False)

print("✓ Saved: random_forest_model.pkl")
print("✓ Saved: y_pred.npy / y_pred_proba.npy")
print("✓ Saved: feature_importance_ranking.csv")

with open(OUTPUT_DIR / "model_results.txt", "w") as f:
    f.write("RANDOM FOREST CLASSIFICATION RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Dataset: UWB LOS/NLOS\n")
    f.write(f"Features: {len(feature_names)} (core + CIR 730-849)\n")
    f.write(f"Training samples: {len(X_train):,}\n")
    f.write(f"Validation samples: {len(X_val):,}\n")
    f.write(f"Test samples    : {len(X_test):,}\n\n")
    f.write("MODEL CONFIGURATION:\n")
    for k, v in best_params.items():
        f.write(f"  - {k}: {v}\n")
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

print("✓ Saved: model_results.txt")

# =============================================================================
# STEP 10: SUMMARY
# =============================================================================
print()
print("=" * 80)
print("RANDOM FOREST CLASSIFICATION - COMPLETE!")
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
    "random_forest_model.pkl",
    "confusion_matrix_and_roc.png",
    "error_analysis_rf.png",
    "feature_importance.png",
    "importance_by_category.png",
    "model_results.txt",
    "error_analysis_rf.txt",
    "feature_importance_ranking.csv",
]:
    print(f"   - {f}")
print()
print("=" * 80)

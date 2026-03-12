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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from pathlib import Path
import joblib
import time
import os
import warnings

warnings.filterwarnings("ignore")

# CONFIGURATION
RUN_NAME = os.getenv("RUN_NAME", "split_env_70_15_15_seed42")
DATA_DIR = Path("./runs") / RUN_NAME / "preprocessed_data"
OUTPUT_DIR = Path("./runs") / RUN_NAME / "models" / "xgboost"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
RUN_TUNING = os.getenv("RUN_TUNING", "1") == "1"
USE_GPU = os.getenv("USE_GPU", "0") == "1"
XGB_DEVICE = "cuda" if USE_GPU else "cpu"
XGB_TREE_METHOD = "hist"

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
    print("Running RandomizedSearchCV (80 iterations, 5-fold CV)...")
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

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

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
        n_iter=80,
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
    print(f"\n✓ Tuning completed in {tuning_time:.2f} seconds")
    print(f"✓ Best CV F1 score : {search.best_score_:.4f}")
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
# STEP 7: FEATURE IMPORTANCE ANALYSIS
# ==============================================================================
print("\nStep 7: Analyzing feature importance...")
print("=" * 80)

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

rf_proba_path = (
    Path("./runs") / RUN_NAME / "models" / "random_forest" / "y_pred_proba.npy"
)
lr_proba_path = Path("./runs") / RUN_NAME / "models" / "logreg_svm" / "y_proba_lr.npy"
svm_proba_path = Path("./runs") / RUN_NAME / "models" / "logreg_svm" / "y_proba_svm.npy"

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
    "feature_importance_xgb.png",
    "importance_by_category_xgb.png",
    "roc_all_models.png",
    "model_results_xgb.txt",
    "feature_importance_xgb.csv",
]:
    print(f"   - {f}")
print()
print("=" * 80)

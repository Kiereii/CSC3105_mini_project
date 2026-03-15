"""
UWB LOS/NLOS Classification - Logistic Regression & SVM Implementation

Why these two algorithms complement Random Forest:
1. Logistic Regression: Linear baseline - fast, interpretable via coefficients,
   tells us which features linearly separate LOS vs NLOS.
2. Support Vector Machine (LinearSVC + RBF probe): Margin-based classifier -
   finds the widest decision boundary, excellent for high-dimensional data
   like our 136-feature space.

Both require StandardScaler - already prepared by preprocess_data.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
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
from pathlib import Path
import joblib
import time
import os
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RUN_NAME = os.getenv("RUN_NAME", "split_env_70_15_15_seed42")
DATA_DIR = Path("./runs") / RUN_NAME / "preprocessed_data"
OUTPUT_DIR = Path("./runs") / RUN_NAME / "models" / "logreg_svm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

print("=" * 80)
print("LOGISTIC REGRESSION & SVM CLASSIFIER - UWB LOS/NLOS PREDICTION")
print("=" * 80)
print()

# ==============================================================================
# STEP 1: LOAD PREPROCESSED DATA
# ==============================================================================
print("Step 1: Loading preprocessed data...")
print("(Using STANDARD-SCALED data - required for LR and SVM)")
print()

X_train = np.load(DATA_DIR / "X_train_standard.npy")
X_test = np.load(DATA_DIR / "X_test_standard.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

# Load feature names using the same parsing logic as random_forest_classifier.py
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

print(f"Training set  : {X_train.shape}")
print(f"Test set      : {X_test.shape}")
print(f"Features      : {len(feature_names)}")
print(f"  LOS  in train : {np.sum(y_train == 0):,}")
print(f"  NLOS in train : {np.sum(y_train == 1):,}")
print()

# ==============================================================================
# HELPER: unified metrics + plots (reused for both models)
# ==============================================================================


def evaluate_model(
    model_name,
    y_true,
    y_pred,
    y_proba,
    cm_save_path,
    feat_imp_path=None,
    feat_names=None,
    feat_values=None,
    feat_label="Importance",
):
    """
    Prints metrics, saves confusion matrix + ROC plot, and optionally
    saves a feature importance / coefficient bar chart.
    Returns a dict of metric values for final comparison.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n {model_name} PERFORMANCE METRICS:")
    print("-" * 40)
    print(f"Accuracy  : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print()
    print(f" DETAILED CLASSIFICATION REPORT ({model_name}):")
    print("-" * 40)
    print(classification_report(y_true, y_pred, target_names=["LOS (0)", "NLOS (1)"]))

    # --- Confusion matrix + ROC curve (side by side, same layout as RF) -------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax1,
        xticklabels=["LOS", "NLOS"],
        yticklabels=["LOS", "NLOS"],
    )
    ax1.set_title(f"Confusion Matrix – {model_name}", fontsize=14, fontweight="bold")
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

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ax2.plot(
        fpr, tpr, color="#e74c3c", linewidth=2, label=f"ROC Curve (AUC = {auc:.4f})"
    )
    ax2.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier")
    ax2.set_xlabel("False Positive Rate", fontsize=12)
    ax2.set_ylabel("True Positive Rate", fontsize=12)
    ax2.set_title(f"ROC Curve – {model_name}", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(cm_save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {cm_save_path.name}")
    plt.close()

    # --- Feature importance / coefficient plot --------------------------------
    if feat_imp_path and feat_names is not None and feat_values is not None:
        top_n = 20
        indices = np.argsort(np.abs(feat_values))[::-1][:top_n]
        top_names = [feat_names[i] for i in indices]
        top_vals = feat_values[indices]

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in top_vals]
        y_pos = np.arange(len(top_names))

        ax.barh(y_pos, top_vals, align="center", color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names)
        ax.invert_yaxis()
        ax.set_xlabel(feat_label, fontsize=12)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(
            f"Top {top_n} Features by {feat_label} – {model_name}",
            fontsize=14,
            fontweight="bold",
        )
        if feat_label == "Coefficient":
            ax.legend(
                handles=[
                    plt.Rectangle((0, 0), 1, 1, color="#e74c3c", label="→ NLOS"),
                    plt.Rectangle((0, 0), 1, 1, color="#3498db", label="→ LOS"),
                ],
                fontsize=10,
            )
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(feat_imp_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {feat_imp_path.name}")
        plt.close()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "cm": cm,
    }


def save_results_txt(path, model_name, config_dict, metrics_dict, feat_top5=None):
    """Saves a model_results_xxx.txt in the same format as the RF script."""
    cm = metrics_dict["cm"]
    with open(path, "w") as f:
        f.write(f"{model_name.upper()} CLASSIFICATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: UWB LOS/NLOS\n")
        f.write(f"Features: {X_train.shape[1]} (16 core + 120 CIR)\n")
        f.write(f"Training samples: {len(X_train):,}\n")
        f.write(f"Test samples: {len(X_test):,}\n\n")
        f.write("MODEL CONFIGURATION:\n")
        for k, v in config_dict.items():
            f.write(f"  - {k}: {v}\n")
        f.write("\nPERFORMANCE METRICS:\n")
        f.write(
            f"  - Accuracy  : {metrics_dict['accuracy']:.4f} "
            f"({metrics_dict['accuracy'] * 100:.2f}%)\n"
        )
        f.write(f"  - Precision : {metrics_dict['precision']:.4f}\n")
        f.write(f"  - Recall    : {metrics_dict['recall']:.4f}\n")
        f.write(f"  - F1-Score  : {metrics_dict['f1']:.4f}\n")
        f.write(f"  - ROC-AUC   : {metrics_dict['roc_auc']:.4f}\n\n")
        f.write("CONFUSION MATRIX:\n")
        f.write(f"  True LOS  predicted as LOS  : {cm[0, 0]:,}\n")
        f.write(f"  True LOS  predicted as NLOS : {cm[0, 1]:,}\n")
        f.write(f"  True NLOS predicted as LOS  : {cm[1, 0]:,}\n")
        f.write(f"  True NLOS predicted as NLOS : {cm[1, 1]:,}\n")
        if feat_top5:
            f.write("\nTOP 5 FEATURES:\n")
            for rank, (name, val) in enumerate(feat_top5, 1):
                f.write(f"  {rank}. {name}: {val:.4f}\n")
    print(f"✓ Saved: {path.name}")


# ==============================================================================
# STEP 2: MODEL 1 – LOGISTIC REGRESSION
# ==============================================================================
print("=" * 80)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 80)
print()
print("Training Logistic Regression...")
print("  solver    : lbfgs  (efficient for medium datasets)")
print("  max_iter  : 1000   (ensure convergence)")
print("  C         : 1.0    (default regularisation)")
print("  class_weight: balanced (handles any slight class imbalance)")
print()

lr_model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    C=1.0,
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

t0 = time.time()
lr_model.fit(X_train, y_train)
lr_time = time.time() - t0
print(f"✓ Training completed in {lr_time:.2f} seconds\n")

# Predictions
y_pred_lr = lr_model.predict(X_test)
y_proba_lr = lr_model.predict_proba(X_test)[:, 1]

# Evaluate
lr_metrics = evaluate_model(
    model_name="Logistic Regression",
    y_true=y_test,
    y_pred=y_pred_lr,
    y_proba=y_proba_lr,
    cm_save_path=OUTPUT_DIR / "confusion_matrix_and_roc_lr.png",
    feat_imp_path=OUTPUT_DIR / "feature_importance_lr.png",
    feat_names=feature_names,
    feat_values=lr_model.coef_[0],  # shape (n_features,)
    feat_label="Coefficient",
)

# Feature importance DataFrame (coefficients)
lr_coef_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Coefficient": lr_model.coef_[0],
        "AbsCoefficient": np.abs(lr_model.coef_[0]),
    }
).sort_values("AbsCoefficient", ascending=False)

lr_coef_df.to_csv(OUTPUT_DIR / "feature_importance_lr.csv", index=False)
print("✓ Saved: feature_importance_lr.csv")

# Top 5 for text file
top5_lr = list(zip(lr_coef_df["Feature"].head(5), lr_coef_df["AbsCoefficient"].head(5)))

save_results_txt(
    path=OUTPUT_DIR / "model_results_lr.txt",
    model_name="Logistic Regression",
    config_dict={
        "solver": "lbfgs",
        "max_iter": 1000,
        "C": 1.0,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
    },
    metrics_dict=lr_metrics,
    feat_top5=top5_lr,
)

# Save model
joblib.dump(lr_model, OUTPUT_DIR / "logistic_regression_model.pkl")
np.save(OUTPUT_DIR / "y_pred_lr.npy", y_pred_lr)
np.save(OUTPUT_DIR / "y_proba_lr.npy", y_proba_lr)
print("✓ Saved: logistic_regression_model.pkl\n")

print("💡 LOGISTIC REGRESSION KEY INSIGHTS:")
print(
    f"   Top feature: {lr_coef_df.iloc[0]['Feature']} "
    f"(|coef| = {lr_coef_df.iloc[0]['AbsCoefficient']:.4f})"
)
print(f"   Positive coef (→ NLOS): {(lr_model.coef_[0] > 0).sum()} features")
print(f"   Negative coef (→ LOS) : {(lr_model.coef_[0] < 0).sum()} features")
print()

# ==============================================================================
# STEP 3: MODEL 2 – SUPPORT VECTOR MACHINE (LinearSVC + probability via calibration)
# ==============================================================================
print("=" * 80)
print("MODEL 2: SUPPORT VECTOR MACHINE (Linear Kernel)")
print("=" * 80)
print()
print("Why LinearSVC instead of RBF SVC?")
print("  • LinearSVC scales to 33k × 136 efficiently (O(n) vs O(n²) for RBF)")
print("  • Still clearly different from Random Forest (margin-based, not tree-based)")
print("  • CalibratedClassifierCV wraps it to give probability scores for ROC curve")
print()
print("Training LinearSVC (C=1.0)...")
print()

# LinearSVC does not output probabilities natively; wrap with calibration
svm_base = LinearSVC(
    C=1.0,
    max_iter=2000,
    random_state=RANDOM_SEED,
    class_weight="balanced",
)

# CalibratedClassifierCV adds probability output via Platt scaling (cv=3)
svm_model = CalibratedClassifierCV(svm_base, cv=3, n_jobs=-1)

t0 = time.time()
svm_model.fit(X_train, y_train)
svm_time = time.time() - t0
print(f"✓ Training completed in {svm_time:.2f} seconds\n")

# Predictions
y_pred_svm = svm_model.predict(X_test)
y_proba_svm = svm_model.predict_proba(X_test)[:, 1]

# Feature importance from the underlying LinearSVC coefficients
# (average over calibration folds)
svm_coefs = np.mean(
    [est.estimator.coef_[0] for est in svm_model.calibrated_classifiers_], axis=0
)

# Evaluate
svm_metrics = evaluate_model(
    model_name="SVM (LinearSVC)",
    y_true=y_test,
    y_pred=y_pred_svm,
    y_proba=y_proba_svm,
    cm_save_path=OUTPUT_DIR / "confusion_matrix_and_roc_svm.png",
    feat_imp_path=OUTPUT_DIR / "feature_importance_svm.png",
    feat_names=feature_names,
    feat_values=svm_coefs,
    feat_label="Coefficient",
)

# Feature importance DataFrame
svm_coef_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Coefficient": svm_coefs,
        "AbsCoefficient": np.abs(svm_coefs),
    }
).sort_values("AbsCoefficient", ascending=False)

svm_coef_df.to_csv(OUTPUT_DIR / "feature_importance_svm.csv", index=False)
print("✓ Saved: feature_importance_svm.csv")

top5_svm = list(
    zip(svm_coef_df["Feature"].head(5), svm_coef_df["AbsCoefficient"].head(5))
)

save_results_txt(
    path=OUTPUT_DIR / "model_results_svm.txt",
    model_name="SVM (LinearSVC + Calibration)",
    config_dict={
        "C": 1.0,
        "max_iter": 2000,
        "class_weight": "balanced",
        "calibration_cv": 3,
        "random_state": RANDOM_SEED,
    },
    metrics_dict=svm_metrics,
    feat_top5=top5_svm,
)

# Save model and predictions
joblib.dump(svm_model, OUTPUT_DIR / "svm_model.pkl")
np.save(OUTPUT_DIR / "y_pred_svm.npy", y_pred_svm)
np.save(OUTPUT_DIR / "y_proba_svm.npy", y_proba_svm)
print("✓ Saved: svm_model.pkl\n")

print("💡 SVM KEY INSIGHTS:")
print(
    f"   Top feature : {svm_coef_df.iloc[0]['Feature']} "
    f"(|coef| = {svm_coef_df.iloc[0]['AbsCoefficient']:.4f})"
)
print(f"   Positive coef (→ NLOS): {(svm_coefs > 0).sum()} features")
print(f"   Negative coef (→ LOS) : {(svm_coefs < 0).sum()} features")
print()

# ==============================================================================
# STEP 4: HEAD-TO-HEAD COMPARISON PLOT (LR vs SVM vs RF on same chart)
# ==============================================================================
print("=" * 80)
print("STEP 4: Combined ROC Comparison Plot (LR vs SVM)")
print("=" * 80)
print()

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(
    fpr_lr,
    tpr_lr,
    color="#e74c3c",
    linewidth=2,
    label=f"Logistic Regression (AUC = {lr_metrics['roc_auc']:.4f})",
)
ax.plot(
    fpr_svm,
    tpr_svm,
    color="#9b59b6",
    linewidth=2,
    label=f"SVM – LinearSVC   (AUC = {svm_metrics['roc_auc']:.4f})",
)
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve Comparison – LR vs SVM", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_comparison_lr_svm.png", dpi=300, bbox_inches="tight")
print("✓ Saved: roc_comparison_lr_svm.png\n")
plt.close()

# ==============================================================================
# STEP 5: FEATURE AGREEMENT ANALYSIS
# Finds features consistently important across LR and SVM
# ==============================================================================
print("=" * 80)
print("STEP 5: Feature Agreement – LR vs SVM")
print("=" * 80)
print()

# Rank by absolute coefficient in each model
lr_ranks = lr_coef_df[["Feature", "AbsCoefficient"]].rename(
    columns={"AbsCoefficient": "LR_AbsCoef"}
)
svm_ranks = svm_coef_df[["Feature", "AbsCoefficient"]].rename(
    columns={"AbsCoefficient": "SVM_AbsCoef"}
)

combined = lr_ranks.merge(svm_ranks, on="Feature")
combined["LR_Rank"] = combined["LR_AbsCoef"].rank(ascending=False).astype(int)
combined["SVM_Rank"] = combined["SVM_AbsCoef"].rank(ascending=False).astype(int)
combined["AvgRank"] = (combined["LR_Rank"] + combined["SVM_Rank"]) / 2
combined = combined.sort_values("AvgRank")

combined.to_csv(OUTPUT_DIR / "feature_agreement_lr_svm.csv", index=False)
print("🏆 TOP 10 FEATURES CONSISTENTLY IMPORTANT ACROSS LR AND SVM:")
print("-" * 60)
print(f"{'Rank':<5} {'Feature':<20} {'LR Rank':<10} {'SVM Rank':<10} {'Avg Rank'}")
print("-" * 60)
for _, row in combined.head(10).iterrows():
    print(
        f"{int(row['AvgRank']):<5} {row['Feature']:<20} "
        f"{row['LR_Rank']:<10} {row['SVM_Rank']:<10} {row['AvgRank']:.1f}"
    )
print()
print("✓ Saved: feature_agreement_lr_svm.csv\n")

# ==============================================================================
# STEP 6: FINAL SUMMARY
# ==============================================================================
print("=" * 80)
print("LOGISTIC REGRESSION & SVM – COMPLETE!")
print("=" * 80)
print()
print(" KEY RESULTS COMPARISON:")
print(f"{'Metric':<15} {'Logistic Regression':<25} {'SVM (LinearSVC)'}")
print("-" * 60)
for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    lr_val = lr_metrics[metric]
    svm_val = svm_metrics[metric]
    better = "← LR better" if lr_val > svm_val else "← SVM better"
    if abs(lr_val - svm_val) < 0.001:
        better = "(≈ equal)"
    print(f"{metric:<15} {lr_val:<25.4f} {svm_val:.4f}  {better}")
print()
print("⏱  Training times:")
print(f"   Logistic Regression : {lr_time:.2f}s")
print(f"   SVM (LinearSVC)     : {svm_time:.2f}s")
print()
print(" Generated files in:", OUTPUT_DIR.absolute())
files = [
    "logistic_regression_model.pkl",
    "svm_model.pkl",
    "confusion_matrix_and_roc_lr.png",
    "confusion_matrix_and_roc_svm.png",
    "feature_importance_lr.png",
    "feature_importance_svm.png",
    "roc_comparison_lr_svm.png",
    "model_results_lr.txt",
    "model_results_svm.txt",
    "feature_importance_lr.csv",
    "feature_importance_svm.csv",
    "feature_agreement_lr_svm.csv",
]
for f in files:
    print(f"   • {f}")
print()
print("=" * 80)
print("✨ NEXT STEPS:")
print("   1. Copy Random Forest's ROC curve onto roc_comparison_lr_svm.png")
print("      (load y_proba from models/random_forest/y_pred_proba.npy)")
print("   2. Compare feature_importance_lr/svm CSVs with RF's")
print("      feature_importance_ranking.csv to find agreement features")
print("   3. Discuss: LR/SVM NLOS recall vs RF — which is safer for positioning?")
print("=" * 80)

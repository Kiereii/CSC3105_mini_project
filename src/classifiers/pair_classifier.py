"""
UWB LOS/NLOS - Pair-Level Classifier
======================================
Objective (from brief):
  "Recognise whether the two shortest dominant paths is either a pair of
   LOS and NLOS or both are NLOS."

This script answers EXACTLY that question at the pair level:
  PAIR_LABEL = 0 → LOS+NLOS  (a trustworthy LOS path exists → use it)
  PAIR_LABEL = 1 → NLOS+NLOS (both paths obstructed → apply bias correction)

Why this is different from the per-path classifiers:
  - Per-path classifiers (RF, XGBoost, LR, SVM) look at Path 1 features ONLY
  - This classifier sees Path 1 AND Path 2 features together (PEAK2_* columns)
  - It answers the positioning system's real question:
    "Do I have a reliable LOS path right now or not?"

Four models trained and compared:
  1. Random Forest      (bagging — no scaling needed)
  2. XGBoost            (boosting — no scaling needed)
  3. Logistic Regression (linear — requires scaling)
  4. SVM                (kernel — requires scaling)

Run AFTER: preprocess_data.py → second_path_features.py
Outputs → models/pair_classifier/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
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
plt.style.use("seaborn-v0_8-darkgrid")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RUN_NAME = os.getenv("RUN_NAME", "split_80_20_seed42")
DATA_DIR = Path("./runs") / RUN_NAME / "preprocessed_data"
OUTPUT_DIR = Path("./runs") / RUN_NAME / "models" / "pair_classifier"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

CLASS_NAMES = ["LOS+NLOS", "NLOS+NLOS"]  # 0, 1
COLORS = {
    "Random Forest": "#2ecc71",
    "XGBoost": "#f39c12",
    "Logistic Regression": "#3498db",
    "SVM": "#9b59b6",
}

print("=" * 80)
print("PAIR-LEVEL CLASSIFIER — LOS+NLOS vs NLOS+NLOS")
print("=" * 80)
print()
print("What this classifies:")
print("  0 = LOS+NLOS  → Path 1 is LOS, Path 2 is NLOS (trustworthy path exists)")
print("  1 = NLOS+NLOS → Both paths obstructed (positioning system must correct)")
print()

# ==============================================================================
# STEP 1: LOAD PAIR DATASET
# ==============================================================================
print("Step 1: Loading pair dataset...")

X_train = np.load(DATA_DIR / "X_train_pair.npy")
X_test = np.load(DATA_DIR / "X_test_pair.npy")
y_train = np.load(DATA_DIR / "y_train_pair.npy")
y_test = np.load(DATA_DIR / "y_test_pair.npy")

# Load feature names
with open(DATA_DIR / "pair_feature_names.txt", "r") as f:
    lines = f.readlines()

feature_names = []
reading = False
for line in lines:
    if line.strip().startswith("Core + Second-Path Features"):
        reading = True
        continue
    if reading and line.strip().startswith(tuple("0123456789")):
        feature_names.append(line.split(".")[-1].strip())
    if line.strip().startswith("CIR Features"):
        for i in range(730, 850):
            feature_names.append(f"CIR{i}")
        break

print(f"  Training samples : {len(X_train):,}")
print(f"  Test samples     : {len(X_test):,}")
print(f"  Features         : {X_train.shape[1]}")
print()
print("  Pair label distribution (test set):")
print(f"    LOS+NLOS  (0): {(y_test == 0).sum():,} ({(y_test == 0).mean() * 100:.1f}%)")
print(f"    NLOS+NLOS (1): {(y_test == 1).sum():,} ({(y_test == 1).mean() * 100:.1f}%)")
print()

# ==============================================================================
# STEP 2: SCALE DATA (for Logistic Regression and SVM)
# ==============================================================================
print("Step 2: Scaling features for LR/SVM...")
print("-" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Scaled {X_train.shape[1]} features (mean=0, std=1)")
print()

# ==============================================================================
# STEP 3: TRAIN RANDOM FOREST
# ==============================================================================
print("Step 2: Training Random Forest pair classifier...")
print("-" * 60)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=1,
)

t0 = time.time()
rf.fit(X_train, y_train)
rf_time = time.time() - t0
print(f"\n✓ RF trained in {rf_time:.1f}s")

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

# ==============================================================================
# STEP 4: TRAIN XGBOOST
# ==============================================================================
print()
print("Step 4: Training XGBoost pair classifier...")
print("-" * 60)

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbosity=1,
)

t0 = time.time()
xgb.fit(X_train, y_train)
xgb_time = time.time() - t0
print(f"✓ XGBoost trained in {xgb_time:.1f}s")

y_pred_xgb = xgb.predict(X_test)
y_proba_xgb = xgb.predict_proba(X_test)[:, 1]

# ==============================================================================
# STEP 5: TRAIN LOGISTIC REGRESSION (scaled data)
# ==============================================================================
print()
print("Step 5: Training Logistic Regression pair classifier...")
print("-" * 60)

lr = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

t0 = time.time()
lr.fit(X_train_scaled, y_train)
lr_time = time.time() - t0
print(f"✓ Logistic Regression trained in {lr_time:.1f}s")

y_pred_lr = lr.predict(X_test_scaled)
y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

# ==============================================================================
# STEP 6: TRAIN SVM (scaled data) - Using LinearSVC for speed
# ==============================================================================
print()
print("Step 6: Training SVM pair classifier...")
print("-" * 60)
print("  (Using LinearSVC with probability calibration - faster than RBF)")

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# LinearSVC is much faster than SVC with RBF kernel
svm_base = LinearSVC(
    C=1.0,
    max_iter=2000,
    random_state=RANDOM_SEED,
    dual=False,  # faster for n_samples > n_features
)

# Calibrate to get probability estimates
svm = CalibratedClassifierCV(svm_base, cv=3, method="sigmoid")

t0 = time.time()
svm.fit(X_train_scaled, y_train)
svm_time = time.time() - t0
print(f"✓ SVM trained in {svm_time:.1f}s")

y_pred_svm = svm.predict(X_test_scaled)
y_proba_svm = svm.predict_proba(X_test_scaled)[:, 1]

# ==============================================================================
# STEP 7: EVALUATE ALL MODELS
# ==============================================================================
print()
print("Step 7: Evaluating all models...")
print("=" * 80)


def evaluate(y_true, y_pred, y_proba, name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    print(f"\n  {name}")
    print(f"    Accuracy  : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    ROC-AUC   : {auc:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
    }


metrics_rf = evaluate(y_test, y_pred_rf, y_proba_rf, "Random Forest")
metrics_xgb = evaluate(y_test, y_pred_xgb, y_proba_xgb, "XGBoost")
metrics_lr = evaluate(y_test, y_pred_lr, y_proba_lr, "Logistic Regression")
metrics_svm = evaluate(y_test, y_pred_svm, y_proba_svm, "SVM")

# ==============================================================================
# STEP 8: CONFUSION MATRICES
# ==============================================================================
print("Step 8: Plotting confusion matrices...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for ax, y_pred_m, name in [
    (axes[0, 0], y_pred_rf, "Random Forest"),
    (axes[0, 1], y_pred_xgb, "XGBoost"),
    (axes[1, 0], y_pred_lr, "Logistic Regression"),
    (axes[1, 1], y_pred_svm, "SVM"),
]:
    cm = confusion_matrix(y_test, y_pred_m)
    total = cm.sum()
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
    ax.set_title(
        f"Pair Classifier — {name}", fontsize=13, fontweight="bold", color=COLORS[name]
    )
    ax.set_xlabel("Predicted Pair", fontsize=11)
    ax.set_ylabel("Actual Pair", fontsize=11)
    for i in range(2):
        for j in range(2):
            ax.text(
                j + 0.5,
                i + 0.75,
                f"({cm[i, j] / total * 100:.1f}%)",
                ha="center",
                va="center",
                fontsize=9,
                color="red",
            )

fig.suptitle(
    "Pair-Level Classification: LOS+NLOS vs NLOS+NLOS", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pair_confusion_matrices.png", dpi=300, bbox_inches="tight")
print("✓ Saved: pair_confusion_matrices.png")
plt.close()

# ==============================================================================
# STEP 9: ROC CURVES
# ==============================================================================
print("Step 9: Plotting ROC curves...")

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(
    [0, 1],
    [0, 1],
    color="gray",
    linestyle="--",
    linewidth=1.5,
    label="Random Classifier (AUC = 0.500)",
)

for y_proba_m, name in [
    (y_proba_rf, "Random Forest"),
    (y_proba_xgb, "XGBoost"),
    (y_proba_lr, "Logistic Regression"),
    (y_proba_svm, "SVM"),
]:
    fpr, tpr, _ = roc_curve(y_test, y_proba_m)
    auc = roc_auc_score(y_test, y_proba_m)
    ax.plot(
        fpr, tpr, color=COLORS[name], linewidth=2.2, label=f"{name} (AUC = {auc:.4f})"
    )

ax.set_xlabel("False Positive Rate\n(NLOS+NLOS misclassified as LOS+NLOS)", fontsize=11)
ax.set_ylabel("True Positive Rate\n(NLOS+NLOS correctly identified)", fontsize=11)
ax.set_title(
    "Pair Classifier ROC Curve\nLOS+NLOS vs NLOS+NLOS", fontsize=13, fontweight="bold"
)
ax.legend(fontsize=10, loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pair_roc_curve.png", dpi=300, bbox_inches="tight")
print("✓ Saved: pair_roc_curve.png")
plt.close()

# ==============================================================================
# STEP 7: FEATURE IMPORTANCE — highlights PEAK2_* features
# ==============================================================================
print("Step 7: Plotting feature importance (with second-path features highlighted)...")

rf_imp = (
    pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": rf.feature_importances_,
        }
    )
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

xgb_imp = (
    pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": xgb.feature_importances_,
        }
    )
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

# ── Top-20 RF importance bar chart (colour-coded by feature type) ──────────────
top20 = rf_imp.head(20).copy()


def feat_color(name):
    if name.startswith("PEAK2"):
        return "#e74c3c"  # red  — second-path features
    elif name.startswith("CIR"):
        return "#3498db"  # blue — CIR waveform features
    else:
        return "#95a5a6"  # grey — core physical features


colors_bar = [feat_color(n) for n in top20["Feature"]]

fig, ax = plt.subplots(figsize=(12, 8))
y_pos = np.arange(len(top20))
ax.barh(y_pos, top20["Importance"], color=colors_bar, alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(top20["Feature"], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Feature Importance (MDI)", fontsize=12)
ax.set_title(
    "Pair Classifier — Top 20 Feature Importance (Random Forest)\n"
    "Red = 2nd-path features  |  Blue = CIR waveform  |  Grey = core",
    fontsize=12,
    fontweight="bold",
)
ax.grid(axis="x", alpha=0.3)

# Legend patches
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="#e74c3c", label="2nd-path features (PEAK2_*)"),
    Patch(facecolor="#3498db", label="CIR waveform features"),
    Patch(facecolor="#95a5a6", label="Core physical features"),
]
ax.legend(handles=legend_elements, fontsize=10, loc="lower right")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pair_feature_importance_rf.png", dpi=300, bbox_inches="tight")
print("✓ Saved: pair_feature_importance_rf.png")
plt.close()

# ── Feature category summary ───────────────────────────────────────────────────
peak2_imp = rf_imp[rf_imp["Feature"].str.startswith("PEAK2")]["Importance"].sum()
cir_imp = rf_imp[rf_imp["Feature"].str.startswith("CIR")]["Importance"].sum()
core_imp = 1.0 - peak2_imp - cir_imp

print(f"\n  Feature importance breakdown (RF):")
print(f"    2nd-path features (PEAK2_*) : {peak2_imp * 100:.1f}%")
print(f"    CIR waveform features       : {cir_imp * 100:.1f}%")
print(f"    Core physical features      : {core_imp * 100:.1f}%")

fig, ax = plt.subplots(figsize=(7, 5))
cats = ["2nd-path\n(PEAK2_*)", "CIR\nwaveform", "Core\nphysical"]
vals = [peak2_imp, cir_imp, core_imp]
cols = ["#e74c3c", "#3498db", "#95a5a6"]
bars = ax.bar(cats, vals, color=cols, alpha=0.85)
for bar, val in zip(bars, vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f"{val * 100:.1f}%",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )
ax.set_ylabel("Total Importance", fontsize=12)
ax.set_ylim(0, max(vals) * 1.2)
ax.set_title(
    "Pair Classifier — Feature Importance by Category (RF)",
    fontsize=12,
    fontweight="bold",
)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "pair_importance_by_category.png", dpi=300, bbox_inches="tight"
)
print("✓ Saved: pair_importance_by_category.png")
plt.close()

# ==============================================================================
# STEP 10: SAVE PREDICTIONS & MODELS
# ==============================================================================
print()
print("Step 10: Saving predictions and models...")

np.save(OUTPUT_DIR / "y_pred_rf.npy", y_pred_rf)
np.save(OUTPUT_DIR / "y_proba_rf.npy", y_proba_rf)
np.save(OUTPUT_DIR / "y_pred_xgb.npy", y_pred_xgb)
np.save(OUTPUT_DIR / "y_proba_xgb.npy", y_proba_xgb)
np.save(OUTPUT_DIR / "y_pred_lr.npy", y_pred_lr)
np.save(OUTPUT_DIR / "y_proba_lr.npy", y_proba_lr)
np.save(OUTPUT_DIR / "y_pred_svm.npy", y_pred_svm)
np.save(OUTPUT_DIR / "y_proba_svm.npy", y_proba_svm)
np.save(OUTPUT_DIR / "y_test_pair.npy", y_test)

joblib.dump(rf, OUTPUT_DIR / "pair_rf_model.pkl", compress=3)
joblib.dump(xgb, OUTPUT_DIR / "pair_xgb_model.pkl", compress=3)
joblib.dump(lr, OUTPUT_DIR / "pair_lr_model.pkl", compress=3)
joblib.dump(svm, OUTPUT_DIR / "pair_svm_model.pkl", compress=3)
joblib.dump(scaler, OUTPUT_DIR / "pair_scaler.pkl", compress=3)

rf_imp.to_csv(OUTPUT_DIR / "pair_feature_importance_rf.csv", index=False)
xgb_imp.to_csv(OUTPUT_DIR / "pair_feature_importance_xgb.csv", index=False)

# ==============================================================================
# STEP 11: SAVE RESULTS TEXT
# ==============================================================================
metrics_df = pd.DataFrame([metrics_rf, metrics_xgb, metrics_lr, metrics_svm]).set_index(
    "Model"
)
metrics_df.to_csv(OUTPUT_DIR / "pair_metrics.csv")

with open(OUTPUT_DIR / "pair_results.txt", "w") as f:
    f.write("PAIR-LEVEL CLASSIFIER RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write("Task: LOS+NLOS vs NLOS+NLOS pair classification\n")
    f.write("Features: 20 core+second-path + 120 CIR = 140 total\n")
    f.write(f"Training samples: {len(X_train):,}\n")
    f.write(f"Test samples    : {len(X_test):,}\n\n")
    f.write("PAIR_LABEL definition:\n")
    f.write("  0 = LOS+NLOS  (Path 1 LOS exists — trustworthy range available)\n")
    f.write("  1 = NLOS+NLOS (Both paths obstructed — apply bias correction)\n\n")
    for name, met in [
        ("Random Forest", metrics_rf),
        ("XGBoost", metrics_xgb),
        ("Logistic Regression", metrics_lr),
        ("SVM", metrics_svm),
    ]:
        f.write(f"{name}:\n")
        f.write(f"  Accuracy  : {met['Accuracy']:.4f}\n")
        f.write(f"  Precision : {met['Precision']:.4f}\n")
        f.write(f"  Recall    : {met['Recall']:.4f}\n")
        f.write(f"  F1-Score  : {met['F1-Score']:.4f}\n")
        f.write(f"  ROC-AUC   : {met['ROC-AUC']:.4f}\n\n")
    f.write("Feature importance breakdown (RF):\n")
    f.write(f"  2nd-path features (PEAK2_*) : {peak2_imp * 100:.1f}%\n")
    f.write(f"  CIR waveform features       : {cir_imp * 100:.1f}%\n")
    f.write(f"  Core physical features      : {core_imp * 100:.1f}%\n")

print("✓ Saved: predictions, models, feature importance CSVs, results.txt")

# ==============================================================================
# STEP 12: FINAL SUMMARY
# ==============================================================================
print()
print("=" * 80)
print("PAIR CLASSIFIER — COMPLETE")
print("=" * 80)
print()
print("RESULTS SUMMARY:")
print(f"  {'Model':<22} {'Accuracy':>10} {'F1-Score':>10} {'ROC-AUC':>10}")
print(f"  {'-' * 54}")
for name, met in [
    ("Random Forest", metrics_rf),
    ("XGBoost", metrics_xgb),
    ("Logistic Regression", metrics_lr),
    ("SVM", metrics_svm),
]:
    print(
        f"  {name:<22} {met['Accuracy']:>10.4f} {met['F1-Score']:>10.4f} {met['ROC-AUC']:>10.4f}"
    )
print()
print("KEY INSIGHT:")
print(f"  PEAK2_* features contribute {peak2_imp * 100:.1f}% of importance")
print(f"  → Including 2nd-path signal geometry helps the pair decision")
print()
print("FILES SAVED TO: models/pair_classifier/")
files = [
    "pair_confusion_matrices.png",
    "pair_roc_curve.png",
    "pair_feature_importance_rf.png",
    "pair_importance_by_category.png",
    "pair_feature_importance_rf.csv",
    "pair_feature_importance_xgb.csv",
    "pair_metrics.csv",
    "pair_results.txt",
    "pair_rf_model.pkl",
    "pair_xgb_model.pkl",
    "pair_lr_model.pkl",
    "pair_svm_model.pkl",
    "pair_scaler.pkl",
    "y_pred_rf.npy / y_proba_rf.npy",
    "y_pred_xgb.npy / y_proba_xgb.npy",
    "y_pred_lr.npy / y_proba_lr.npy",
    "y_pred_svm.npy / y_proba_svm.npy",
]
for fn in files:
    print(f"  • {fn}")
print()
print("NEXT: Run compare_models.py to include pair results in the unified report")
print("=" * 80)

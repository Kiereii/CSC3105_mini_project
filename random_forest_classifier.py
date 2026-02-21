"""
UWB LOS/NLOS Classification - Random Forest Implementation
Random Forest is perfect for this problem because:
1. No scaling needed (use unscaled data)
2. Shows feature importance (which of your 136 features matter most!)
3. Handles non-linear relationships
4. Robust to outliers
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
from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = Path("./preprocessed_data")
OUTPUT_DIR = Path("./models/random_forest")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

print("=" * 80)
print("RANDOM FOREST CLASSIFIER - UWB LOS/NLOS PREDICTION")
print("=" * 80)
print()

# =============================================================================
# STEP 1: LOAD PREPROCESSED DATA
# =============================================================================
print("Step 1: Loading preprocessed data...")
print("(Using UNSCALED data - Random Forest doesn't need scaling)")
print()

X_train = np.load(DATA_DIR / "X_train_unscaled.npy")
X_test = np.load(DATA_DIR / "X_test_unscaled.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
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
            # Add CIR feature names
            for i in range(730, 850):
                feature_names.append(f"CIR{i}")
            break

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Number of features: {len(feature_names)}")
print(f"  - LOS samples in training: {np.sum(y_train == 0):,}")
print(f"  - NLOS samples in training: {np.sum(y_train == 1):,}")
print()

# =============================================================================
# STEP 2: TRAIN RANDOM FOREST MODEL
# =============================================================================
print("Step 2: Training Random Forest model...")
print("-" * 80)

# Create Random Forest classifier
# Parameters explained:
# - n_estimators: Number of trees (100 is a good default)
# - max_depth: Maximum depth of trees (None = grow until pure)
# - random_state: For reproducibility
# - n_jobs: Use all CPU cores for faster training

rf_model = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    max_depth=None,  # Let trees grow fully
    min_samples_split=2,  # Minimum samples to split a node
    min_samples_leaf=1,  # Minimum samples in leaf node
    random_state=RANDOM_SEED,  # Reproducibility
    n_jobs=-1,  # Use all CPU cores
    verbose=1,  # Show progress
)

# Train the model
start_time = time.time()
rf_model.fit(X_train, y_train)
training_time = time.time() - start_time

print(f"\n✓ Training completed in {training_time:.2f} seconds")
print(f"✓ Number of trees: {rf_model.n_estimators}")
print()

# =============================================================================
# STEP 3: MAKE PREDICTIONS
# =============================================================================
print("Step 3: Making predictions...")

# Predict on test set
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # Probability of NLOS (class 1)

print(f"✓ Predictions completed")
print()

# =============================================================================
# STEP 4: EVALUATE MODEL PERFORMANCE
# =============================================================================
print("Step 4: Evaluating model performance...")
print("=" * 80)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n📊 PERFORMANCE METRICS:")
print("-" * 40)
print(f"Accuracy:  {accuracy:.4f}  ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {auc:.4f}")
print()

# Detailed classification report
print("📋 DETAILED CLASSIFICATION REPORT:")
print("-" * 40)
target_names = ["LOS (0)", "NLOS (1)"]
print(classification_report(y_test, y_pred, target_names=target_names))

# =============================================================================
# STEP 5: CONFUSION MATRIX
# =============================================================================
print("Step 5: Creating confusion matrix visualization...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax1,
    xticklabels=["LOS", "NLOS"],
    yticklabels=["LOS", "NLOS"],
)
ax1.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")

# Add percentage annotations
total = cm.sum()
for i in range(2):
    for j in range(2):
        percentage = cm[i, j] / total * 100
        ax1.text(
            j + 0.5,
            i + 0.7,
            f"({percentage:.1f}%)",
            ha="center",
            va="center",
            fontsize=10,
            color="red",
        )

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
ax2.plot(fpr, tpr, color="#2ecc71", linewidth=2, label=f"ROC Curve (AUC = {auc:.4f})")
ax2.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier")
ax2.set_xlabel("False Positive Rate", fontsize=12)
ax2.set_ylabel("True Positive Rate", fontsize=12)
ax2.set_title("ROC Curve", fontsize=14, fontweight="bold")
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix_and_roc.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: confusion_matrix_and_roc.png")
plt.close()

# =============================================================================
# STEP 6: FEATURE IMPORTANCE ANALYSIS
# =============================================================================
print("Step 6: Analyzing feature importance...")
print("=" * 80)

# Get feature importances
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]  # Sort in descending order

# Create feature importance dataframe
feature_importance_df = pd.DataFrame(
    {"Feature": [feature_names[i] for i in indices], "Importance": importances[indices]}
)

# Display top 20 features
print("\n🏆 TOP 20 MOST IMPORTANT FEATURES:")
print("-" * 50)
for i in range(min(20, len(feature_names))):
    feat_name = feature_names[indices[i]]
    importance = importances[indices[i]]
    print(f"{i + 1:2d}. {feat_name:15s}: {importance:.4f}")
print()

# Visualize top 20 features
fig, ax = plt.subplots(figsize=(12, 8))
top_n = 20
top_indices = indices[:top_n]
top_features = [feature_names[i] for i in top_indices]
top_importances = importances[top_indices]

y_pos = np.arange(len(top_features))
ax.barh(y_pos, top_importances, align="center", color="#3498db")
ax.set_yticks(y_pos)
ax.set_yticklabels(top_features)
ax.invert_yaxis()  # Labels read top-to-bottom
ax.set_xlabel("Feature Importance", fontsize=12)
ax.set_title(
    f"Top {top_n} Most Important Features (Random Forest)",
    fontsize=14,
    fontweight="bold",
)
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: feature_importance.png")
plt.close()

# =============================================================================
# STEP 7: ANALYZE FEATURE CATEGORIES
# =============================================================================
print("Step 7: Analyzing feature categories...")

# Separate importances by feature type
core_importance = []
cir_importance = []

for i, feat_name in enumerate(feature_names):
    if feat_name.startswith("CIR"):
        cir_importance.append(importances[i])
    else:
        core_importance.append(importances[i])

core_total = np.sum(core_importance)
cir_total = np.sum(cir_importance)

print(f"\n📊 IMPORTANCE BY CATEGORY:")
print("-" * 40)
print(f"Core Features (16):  {core_total:.4f} ({core_total * 100:.1f}%)")
print(f"CIR Features (120):  {cir_total:.4f} ({cir_total * 100:.1f}%)")
print()

# Visualize category comparison
fig, ax = plt.subplots(figsize=(8, 6))
categories = ["Core Features\n(16 features)", "CIR Features\n(120 features)"]
values = [core_total, cir_total]
colors = ["#e74c3c", "#3498db"]

bars = ax.bar(categories, values, color=colors, alpha=0.8)
ax.set_ylabel("Total Importance", fontsize=12)
ax.set_title("Feature Importance by Category", fontsize=14, fontweight="bold")
ax.set_ylim(0, 1)

# Add value labels on bars
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
print(f"✓ Saved: importance_by_category.png")
plt.close()

# =============================================================================
# STEP 8: SAVE RESULTS
# =============================================================================
print("Step 8: Saving results...")
print("=" * 80)

# Save model
import joblib

joblib.dump(rf_model, OUTPUT_DIR / "random_forest_model.pkl")
print(f"✓ Saved: random_forest_model.pkl")

# Save predictions
np.save(OUTPUT_DIR / "y_pred.npy", y_pred)
np.save(OUTPUT_DIR / "y_pred_proba.npy", y_pred_proba)
print(f"✓ Saved: predictions")

# Save metrics to file
with open(OUTPUT_DIR / "model_results.txt", "w") as f:
    f.write("RANDOM FOREST CLASSIFICATION RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Dataset: UWB LOS/NLOS\n")
    f.write(f"Features: {len(feature_names)} (16 core + 120 CIR)\n")
    f.write(f"Training samples: {len(X_train):,}\n")
    f.write(f"Test samples: {len(X_test):,}\n\n")

    f.write("MODEL CONFIGURATION:\n")
    f.write(f"  - Number of trees: {rf_model.n_estimators}\n")
    f.write(f"  - Max depth: {rf_model.max_depth}\n")
    f.write(f"  - Random seed: {RANDOM_SEED}\n\n")

    f.write("PERFORMANCE METRICS:\n")
    f.write(f"  - Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
    f.write(f"  - Precision: {precision:.4f}\n")
    f.write(f"  - Recall:    {recall:.4f}\n")
    f.write(f"  - F1-Score:  {f1:.4f}\n")
    f.write(f"  - ROC-AUC:   {auc:.4f}\n\n")

    f.write("FEATURE IMPORTANCE SUMMARY:\n")
    f.write(
        f"  - Core features importance: {core_total:.4f} ({core_total * 100:.1f}%)\n"
    )
    f.write(
        f"  - CIR features importance:  {cir_total:.4f} ({cir_total * 100:.1f}%)\n\n"
    )

    f.write("CONFUSION MATRIX:\n")
    f.write(f"  True LOS predicted as LOS:   {cm[0, 0]:,}\n")
    f.write(f"  True LOS predicted as NLOS:  {cm[0, 1]:,}\n")
    f.write(f"  True NLOS predicted as LOS:  {cm[1, 0]:,}\n")
    f.write(f"  True NLOS predicted as NLOS: {cm[1, 1]:,}\n")

# Save top features to CSV
feature_importance_df.to_csv(OUTPUT_DIR / "feature_importance_ranking.csv", index=False)
print(f"✓ Saved: feature_importance_ranking.csv")

print(f"\n✓ All results saved to: {OUTPUT_DIR}")
print()

# =============================================================================
# STEP 9: SUMMARY
# =============================================================================
print("=" * 80)
print("RANDOM FOREST CLASSIFICATION - COMPLETE!")
print("=" * 80)
print()
print("🎯 KEY RESULTS:")
print(f"   Accuracy:  {accuracy * 100:.2f}%")
print(f"   F1-Score:  {f1:.4f}")
print(f"   ROC-AUC:   {auc:.4f}")
print()
print("💡 KEY INSIGHTS:")
print(
    f"   • Top feature: {feature_names[indices[0]]} (importance: {importances[indices[0]]:.4f})"
)
print(f"   • Core features contribute {core_total * 100:.1f}% of total importance")
print(f"   • CIR features contribute {cir_total * 100:.1f}% of total importance")
print()
print("📁 Generated Files:")
print(f"   • {OUTPUT_DIR}/random_forest_model.pkl")
print(f"   • {OUTPUT_DIR}/confusion_matrix_and_roc.png")
print(f"   • {OUTPUT_DIR}/feature_importance.png")
print(f"   • {OUTPUT_DIR}/importance_by_category.png")
print(f"   • {OUTPUT_DIR}/model_results.txt")
print(f"   • {OUTPUT_DIR}/feature_importance_ranking.csv")
print()
print("=" * 80)
print()
print("✨ NEXT STEPS FOR YOUR TEAM:")
print("   1. Compare these results with other algorithms (Logistic Regression, SVM)")
print("   2. Analyze which features consistently appear as important across models")
print("   3. Consider feature selection based on these importance rankings")
print("   4. Build regression model for distance estimation")
print()

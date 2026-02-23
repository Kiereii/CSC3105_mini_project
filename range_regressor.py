"""
UWB LOS/NLOS - Range Estimator (Regression)
============================================
Part 2 of the project brief:
  "Predict the measured range for the two dominant shortest paths"

Uses Random Forest Regressor (consistent with the classifier choice).
Evaluates with RMSE, MAE, and R² for both Path 1 and Path 2.

Run second_path_features.py FIRST to generate the input .npy files.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import joblib
import time
import warnings

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR   = Path("./preprocessed_data")
OUTPUT_DIR = Path("./models/range_regressor")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

print("=" * 80)
print("RANDOM FOREST RANGE ESTIMATOR — PATH 1 & PATH 2")
print("=" * 80)
print()

# ── Load data ──────────────────────────────────────────────────────────────────
print("Step 1: Loading regression data...")

X_train    = np.load(DATA_DIR / "X_train_regression.npy")
X_test     = np.load(DATA_DIR / "X_test_regression.npy")
y_p1_train = np.load(DATA_DIR / "y_range_p1_train.npy")
y_p1_test  = np.load(DATA_DIR / "y_range_p1_test.npy")
y_p2_train = np.load(DATA_DIR / "y_range_p2_train.npy")
y_p2_test  = np.load(DATA_DIR / "y_range_p2_test.npy")

with open(DATA_DIR / "regression_feature_names.txt") as f:
    lines = f.readlines()

print(f"  Training samples : {len(X_train):,}")
print(f"  Test samples     : {len(X_test):,}")
print(f"  Features         : {X_train.shape[1]}")
print(f"  Path 1 range     : {y_p1_train.min():.2f} – {y_p1_train.max():.2f} m")
print(f"  Path 2 range     : {y_p2_train.min():.2f} – {y_p2_train.max():.2f} m")
print()

# ── Helper: evaluate and report ────────────────────────────────────────────────
def evaluate(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  {label}")
    print(f"    RMSE : {rmse:.4f} m")
    print(f"    MAE  : {mae:.4f} m")
    print(f"    R²   : {r2:.4f}")
    return rmse, mae, r2

# =============================================================================
# STEP 2: PATH 1 RANGE ESTIMATOR
# =============================================================================
print("Step 2: Training Path 1 range estimator...")
print("-" * 60)

rf_p1 = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=0,
)

t0 = time.time()
rf_p1.fit(X_train, y_p1_train)
print(f"  ✓ Trained in {time.time() - t0:.1f}s")

y_p1_pred = rf_p1.predict(X_test)
p1_rmse, p1_mae, p1_r2 = evaluate(y_p1_test, y_p1_pred, "Path 1 Performance")
print()

# =============================================================================
# STEP 3: PATH 2 RANGE ESTIMATOR
# =============================================================================
print("Step 3: Training Path 2 range estimator...")
print("-" * 60)

rf_p2 = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=0,
)

t0 = time.time()
rf_p2.fit(X_train, y_p2_train)
print(f"  ✓ Trained in {time.time() - t0:.1f}s")

y_p2_pred = rf_p2.predict(X_test)
p2_rmse, p2_mae, p2_r2 = evaluate(y_p2_test, y_p2_pred, "Path 2 Performance")
print()

# =============================================================================
# STEP 4: VISUALIZATIONS
# =============================================================================
print("Step 4: Generating visualizations...")

plt.style.use("seaborn-v0_8-darkgrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Range Estimation Results — Path 1 & Path 2", fontsize=16, fontweight="bold")

# ── Row 0: Path 1 ──────────────────────────────────────────────────────────────
# 0,0 — Predicted vs Actual scatter
ax = axes[0, 0]
ax.scatter(y_p1_test, y_p1_pred, alpha=0.3, s=8, color="#3498db", label="Predictions")
lims = [min(y_p1_test.min(), y_p1_pred.min()),
        max(y_p1_test.max(), y_p1_pred.max())]
ax.plot(lims, lims, "r--", linewidth=2, label="Perfect fit")
ax.set_xlabel("Actual Range (m)")
ax.set_ylabel("Predicted Range (m)")
ax.set_title(f"Path 1 — Predicted vs Actual\nR²={p1_r2:.4f}  RMSE={p1_rmse:.4f}m",
             fontweight="bold")
ax.legend(fontsize=9)

# 0,1 — Residuals
ax = axes[0, 1]
residuals_p1 = y_p1_pred - y_p1_test
ax.hist(residuals_p1, bins=60, color="#3498db", alpha=0.8, edgecolor="white")
ax.axvline(0, color="red", linestyle="--", linewidth=2)
ax.set_xlabel("Residual (m)")
ax.set_ylabel("Count")
ax.set_title(f"Path 1 — Residual Distribution\nMAE={p1_mae:.4f}m", fontweight="bold")

# 0,2 — Absolute error vs actual range
ax = axes[0, 2]
ax.scatter(y_p1_test, np.abs(residuals_p1), alpha=0.3, s=8, color="#3498db")
ax.axhline(p1_mae, color="red", linestyle="--", linewidth=2, label=f"MAE={p1_mae:.3f}m")
ax.set_xlabel("Actual Range (m)")
ax.set_ylabel("|Error| (m)")
ax.set_title("Path 1 — Absolute Error vs Range", fontweight="bold")
ax.legend(fontsize=9)

# ── Row 1: Path 2 ──────────────────────────────────────────────────────────────
# 1,0 — Predicted vs Actual scatter
ax = axes[1, 0]
ax.scatter(y_p2_test, y_p2_pred, alpha=0.3, s=8, color="#e74c3c", label="Predictions")
lims = [min(y_p2_test.min(), y_p2_pred.min()),
        max(y_p2_test.max(), y_p2_pred.max())]
ax.plot(lims, lims, "b--", linewidth=2, label="Perfect fit")
ax.set_xlabel("Actual Range (m)")
ax.set_ylabel("Predicted Range (m)")
ax.set_title(f"Path 2 — Predicted vs Actual\nR²={p2_r2:.4f}  RMSE={p2_rmse:.4f}m",
             fontweight="bold")
ax.legend(fontsize=9)

# 1,1 — Residuals
ax = axes[1, 1]
residuals_p2 = y_p2_pred - y_p2_test
ax.hist(residuals_p2, bins=60, color="#e74c3c", alpha=0.8, edgecolor="white")
ax.axvline(0, color="blue", linestyle="--", linewidth=2)
ax.set_xlabel("Residual (m)")
ax.set_ylabel("Count")
ax.set_title(f"Path 2 — Residual Distribution\nMAE={p2_mae:.4f}m", fontweight="bold")

# 1,2 — Side-by-side metrics comparison
ax = axes[1, 2]
metrics = ["RMSE (m)", "MAE (m)", "R²"]
p1_vals = [p1_rmse, p1_mae, p1_r2]
p2_vals = [p2_rmse, p2_mae, p2_r2]
x = np.arange(len(metrics))
width = 0.35
bars1 = ax.bar(x - width/2, p1_vals, width, label="Path 1", color="#3498db", alpha=0.85)
bars2 = ax.bar(x + width/2, p2_vals, width, label="Path 2", color="#e74c3c", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_title("Path 1 vs Path 2 — Metrics Comparison", fontweight="bold")
ax.legend()
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "range_estimation_results.png", dpi=300, bbox_inches="tight")
print(f"  ✓ Saved: range_estimation_results.png")
plt.close()

# ── Feature importance (Path 1) ────────────────────────────────────────────────
print("  Plotting feature importance...")

# Read feature names
core_feats = [
    "FP_IDX", "FP_AMP1", "FP_AMP2", "FP_AMP3",
    "STDEV_NOISE", "CIR_PWR", "MAX_NOISE", "RXPACC",
    "CH", "FRAME_LEN", "PREAM_LEN", "BITRATE", "PRFR",
    "SNR", "SNR_dB",
    "PEAK2_IDX", "PEAK2_AMP", "PEAK2_GAP", "PEAK2_FOUND",
]
cir_feats   = [f"CIR{i}" for i in range(730, 850)]
feat_names  = core_feats + cir_feats

importances_p1 = rf_p1.feature_importances_
importances_p2 = rf_p2.feature_importances_

top_n   = 20
idx_p1  = np.argsort(importances_p1)[::-1][:top_n]
idx_p2  = np.argsort(importances_p2)[::-1][:top_n]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, importances, indices, title, color in [
    (axes[0], importances_p1, idx_p1, "Path 1 — Top 20 Feature Importances", "#3498db"),
    (axes[1], importances_p2, idx_p2, "Path 2 — Top 20 Feature Importances", "#e74c3c"),
]:
    y_pos = np.arange(top_n)
    ax.barh(y_pos, importances[indices], color=color, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feat_names[i] for i in indices], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.grid(axis="x", alpha=0.3)

plt.suptitle("Feature Importance — Range Regressors", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "regressor_feature_importance.png", dpi=300, bbox_inches="tight")
print(f"  ✓ Saved: regressor_feature_importance.png")
plt.close()

# ── Two-path pair classification summary ──────────────────────────────────────
# Per brief: determine if pair is LOS+NLOS or NLOS+NLOS
y_p1_class = np.load(DATA_DIR / "y_test.npy")   # reuse classifier test labels

los_nlos_count  = np.sum(y_p1_class == 0)   # LOS first path → pair = LOS+NLOS
nlos_nlos_count = np.sum(y_p1_class == 1)   # NLOS first path → pair = NLOS+NLOS
total = len(y_p1_class)

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    ["LOS + NLOS\n(Path1=LOS)", "NLOS + NLOS\n(Path1=NLOS)"],
    [los_nlos_count, nlos_nlos_count],
    color=["#2ecc71", "#e74c3c"],
    alpha=0.85,
    edgecolor="white"
)
for bar, cnt in zip(bars, [los_nlos_count, nlos_nlos_count]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f"{cnt:,}\n({cnt/total*100:.1f}%)",
            ha="center", va="bottom", fontweight="bold", fontsize=11)
ax.set_ylabel("Number of Samples")
ax.set_title("Two-Path Pair Classification\n(Path 2 is always NLOS per brief)",
             fontweight="bold", fontsize=13)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "two_path_pair_distribution.png", dpi=300, bbox_inches="tight")
print(f"  ✓ Saved: two_path_pair_distribution.png")
plt.close()

# =============================================================================
# STEP 5: SAVE MODELS AND RESULTS
# =============================================================================
print()
print("Step 5: Saving models and results...")

joblib.dump(rf_p1, OUTPUT_DIR / "rf_range_path1.pkl")
joblib.dump(rf_p2, OUTPUT_DIR / "rf_range_path2.pkl")
np.save(OUTPUT_DIR / "y_p1_pred.npy", y_p1_pred)
np.save(OUTPUT_DIR / "y_p2_pred.npy", y_p2_pred)

with open(OUTPUT_DIR / "regression_results.txt", "w") as f:
    f.write("RANGE ESTIMATION RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Training samples : {len(X_train):,}\n")
    f.write(f"Test samples     : {len(X_test):,}\n")
    f.write(f"Features         : {X_train.shape[1]}\n\n")

    f.write("PATH 1 RANGE ESTIMATOR:\n")
    f.write(f"  RMSE : {p1_rmse:.4f} m\n")
    f.write(f"  MAE  : {p1_mae:.4f} m\n")
    f.write(f"  R²   : {p1_r2:.4f}\n\n")

    f.write("PATH 2 RANGE ESTIMATOR:\n")
    f.write(f"  RMSE : {p2_rmse:.4f} m\n")
    f.write(f"  MAE  : {p2_mae:.4f} m\n")
    f.write(f"  R²   : {p2_r2:.4f}\n\n")

    f.write("TWO-PATH PAIR CLASSIFICATION:\n")
    f.write(f"  LOS + NLOS  pairs : {los_nlos_count:,} ({los_nlos_count/total*100:.1f}%)\n")
    f.write(f"  NLOS + NLOS pairs : {nlos_nlos_count:,} ({nlos_nlos_count/total*100:.1f}%)\n\n")

    f.write("NOTE: Path 2 is always classified as NLOS per project brief.\n")
    f.write("      Range for Path 2 is estimated from second CIR peak offset.\n")

print(f"  ✓ rf_range_path1.pkl")
print(f"  ✓ rf_range_path2.pkl")
print(f"  ✓ regression_results.txt")
print()

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 80)
print("RANGE ESTIMATION COMPLETE!")
print("=" * 80)
print()
print("📊 RESULTS SUMMARY:")
print(f"   Path 1 — RMSE: {p1_rmse:.4f}m  |  MAE: {p1_mae:.4f}m  |  R²: {p1_r2:.4f}")
print(f"   Path 2 — RMSE: {p2_rmse:.4f}m  |  MAE: {p2_mae:.4f}m  |  R²: {p2_r2:.4f}")
print()
print("📁 Output files in:", OUTPUT_DIR.absolute())
print("   • range_estimation_results.png")
print("   • regressor_feature_importance.png")
print("   • two_path_pair_distribution.png")
print("   • rf_range_path1.pkl / rf_range_path2.pkl")
print("   • regression_results.txt")
print()
print("▶  NEXT: Run eda_focused.py to visualise second-path peak detections")
print("=" * 80)


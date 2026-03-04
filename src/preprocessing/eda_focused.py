"""
UWB LOS/NLOS Dataset - Enhanced EDA with Focused CIR Analysis
Focuses on the region of interest around FP_IDX (730-850) for better signal analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Configuration
DATA_DIR = Path("./Dataset/UWB-LOS-NLOS-Data-Set/dataset/Cleaned")
OUTPUT_DIR = Path("./eda_output_focused")
OUTPUT_DIR.mkdir(exist_ok=True)

# Focus region around first path
FOCUS_START = 730
FOCUS_END = 850
CIR_COLUMNS = [f"CIR{i}" for i in range(1016)]
FOCUS_COLUMNS = [f"CIR{i}" for i in range(FOCUS_START, FOCUS_END)]

print("=" * 80)
print("UWB LOS/NLOS DATASET - FOCUSED EDA (Region: 730-850)")
print("=" * 80)
print()

# Load data
print("Loading dataset...")
csv_files = sorted(DATA_DIR.glob("uwb_cleaned_dataset_part*.csv"))
df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
print(f"✓ Loaded {len(df):,} samples")
print()

# Separate classes
los_df = df[df["NLOS"] == 0]
nlos_df = df[df["NLOS"] == 1]

print(f"LOS samples: {len(los_df):,}")
print(f"NLOS samples: {len(nlos_df):,}")
print()

# =============================================================================
# 1. FOCUSED CIR SAMPLES COMPARISON
# =============================================================================
print("Creating focused CIR sample plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
np.random.seed(42)

# Plot individual samples in focused region
for idx in range(3):
    # LOS samples
    ax = axes[0, idx]
    sample = los_df.sample(1, random_state=42 + idx).iloc[0]
    cir_data = sample[CIR_COLUMNS]

    ax.plot(range(1016), cir_data, color="#2ecc71", linewidth=1.2, alpha=0.8)
    ax.axvline(
        x=sample["FP_IDX"],
        color="darkgreen",
        linestyle="--",
        linewidth=2,
        label=f"FP_IDX={sample['FP_IDX']:.0f}",
    )
    ax.set_xlim(FOCUS_START, FOCUS_END)
    ax.set_title(f"LOS Sample {idx + 1}", fontweight="bold", fontsize=12)
    ax.set_xlabel("CIR Sample Index")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # NLOS samples
    ax = axes[1, idx]
    sample = nlos_df.sample(1, random_state=42 + idx).iloc[0]
    cir_data = sample[CIR_COLUMNS]

    ax.plot(range(1016), cir_data, color="#e74c3c", linewidth=1.2, alpha=0.8)
    ax.axvline(
        x=sample["FP_IDX"],
        color="darkred",
        linestyle="--",
        linewidth=2,
        label=f"FP_IDX={sample['FP_IDX']:.0f}",
    )
    ax.set_xlim(FOCUS_START, FOCUS_END)
    ax.set_title(f"NLOS Sample {idx + 1}", fontweight="bold", fontsize=12)
    ax.set_xlabel("CIR Sample Index")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

plt.suptitle(
    f"Individual CIR Samples - Focused View ({FOCUS_START}-{FOCUS_END})",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "focused_cir_samples.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: focused_cir_samples.png")
plt.close()

# =============================================================================
# 2. MEAN CIR PROFILES - FOCUSED
# =============================================================================
print("Creating mean CIR profile comparison...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Full range
los_mean_full = los_df[CIR_COLUMNS].mean()
nlos_mean_full = nlos_df[CIR_COLUMNS].mean()

axes[0].plot(los_mean_full.values, label="LOS Mean", color="#2ecc71", linewidth=2)
axes[0].plot(nlos_mean_full.values, label="NLOS Mean", color="#e74c3c", linewidth=2)
axes[0].axvline(x=745, color="gray", linestyle=":", alpha=0.5, label="FP_IDX region")
axes[0].set_title(
    "Mean CIR Profiles - Full Range (0-1015)", fontweight="bold", fontsize=13
)
axes[0].set_xlabel("CIR Sample Index")
axes[0].set_ylabel("Amplitude")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_xlim(0, 1015)

# Focused range
los_mean_focused = los_df[FOCUS_COLUMNS].mean()
nlos_mean_focused = nlos_df[FOCUS_COLUMNS].mean()

axes[1].plot(
    range(FOCUS_START, FOCUS_END),
    los_mean_focused.values,
    label="LOS Mean",
    color="#2ecc71",
    linewidth=2.5,
)
axes[1].plot(
    range(FOCUS_START, FOCUS_END),
    nlos_mean_focused.values,
    label="NLOS Mean",
    color="#e74c3c",
    linewidth=2.5,
)
axes[1].fill_between(
    range(FOCUS_START, FOCUS_END), los_mean_focused.values, alpha=0.3, color="#2ecc71"
)
axes[1].fill_between(
    range(FOCUS_START, FOCUS_END), nlos_mean_focused.values, alpha=0.3, color="#e74c3c"
)
axes[1].set_title(
    f"Mean CIR Profiles - Focused ({FOCUS_START}-{FOCUS_END})",
    fontweight="bold",
    fontsize=13,
)
axes[1].set_xlabel("CIR Sample Index")
axes[1].set_ylabel("Amplitude")
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_xticks(range(FOCUS_START, FOCUS_END + 1, 20))

plt.suptitle(
    "CIR Profile Comparison: Full vs Focused View", fontsize=15, fontweight="bold"
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mean_cir_comparison.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: mean_cir_comparison.png")
plt.close()

# =============================================================================
# 3. FIRST PATH ANALYSIS
# =============================================================================
print("Analyzing first path characteristics...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# FP_IDX distribution
axes[0, 0].hist(
    los_df["FP_IDX"], bins=50, alpha=0.7, label="LOS", color="#2ecc71", density=True
)
axes[0, 0].hist(
    nlos_df["FP_IDX"], bins=50, alpha=0.7, label="NLOS", color="#e74c3c", density=True
)
axes[0, 0].axvline(
    x=745, color="black", linestyle="--", linewidth=2, label="Avg FP_IDX"
)
axes[0, 0].set_title("First Path Index (FP_IDX) Distribution", fontweight="bold")
axes[0, 0].set_xlabel("FP_IDX")
axes[0, 0].set_ylabel("Density")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# First path amplitudes
axes[0, 1].hist(
    los_df["FP_AMP1"], bins=50, alpha=0.6, label="LOS", color="#2ecc71", density=True
)
axes[0, 1].hist(
    nlos_df["FP_AMP1"], bins=50, alpha=0.6, label="NLOS", color="#e74c3c", density=True
)
axes[0, 1].set_title("FP_AMP1 Distribution", fontweight="bold")
axes[0, 1].set_xlabel("FP_AMP1")
axes[0, 1].set_ylabel("Density")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# FP_IDX vs Range scatter
scatter1 = axes[1, 0].scatter(
    los_df["FP_IDX"], los_df["RANGE"], c="#2ecc71", alpha=0.5, s=10, label="LOS"
)
scatter2 = axes[1, 0].scatter(
    nlos_df["FP_IDX"], nlos_df["RANGE"], c="#e74c3c", alpha=0.5, s=10, label="NLOS"
)
axes[1, 0].set_title("FP_IDX vs Measured Range", fontweight="bold")
axes[1, 0].set_xlabel("FP_IDX")
axes[1, 0].set_ylabel("Range (m)")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Amplitude comparison (boxplot)
data_to_plot = [
    los_df["FP_AMP1"],
    nlos_df["FP_AMP1"],
    los_df["FP_AMP2"],
    nlos_df["FP_AMP2"],
]
bp = axes[1, 1].boxplot(
    data_to_plot,
    labels=["LOS\nAMP1", "NLOS\nAMP1", "LOS\nAMP2", "NLOS\nAMP2"],
    patch_artist=True,
)
colors = ["#2ecc71", "#e74c3c", "#2ecc71", "#e74c3c"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1, 1].set_title("First Path Amplitudes Comparison", fontweight="bold")
axes[1, 1].set_ylabel("Amplitude")
axes[1, 1].grid(axis="y", alpha=0.3)

plt.suptitle("First Path Analysis", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "first_path_analysis.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: first_path_analysis.png")
plt.close()

# =============================================================================
# 4. CIR ENERGY ANALYSIS IN FOCUSED REGION
# =============================================================================
print("Computing CIR energy in focused region...")

# Calculate energy in focused window
df["CIR_ENERGY_FOCUS"] = df[FOCUS_COLUMNS].sum(axis=1)
df["CIR_PEAK_FOCUS"] = df[FOCUS_COLUMNS].max(axis=1)
df["CIR_PEAK_IDX_FOCUS"] = (
    df[FOCUS_COLUMNS].idxmax(axis=1).str.replace("CIR", "").astype(int)
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Energy distribution
axes[0].hist(
    df[df["NLOS"] == 0]["CIR_ENERGY_FOCUS"],
    bins=50,
    alpha=0.7,
    label="LOS",
    color="#2ecc71",
    density=True,
)
axes[0].hist(
    df[df["NLOS"] == 1]["CIR_ENERGY_FOCUS"],
    bins=50,
    alpha=0.7,
    label="NLOS",
    color="#e74c3c",
    density=True,
)
axes[0].set_title(f"CIR Energy in Region {FOCUS_START}-{FOCUS_END}", fontweight="bold")
axes[0].set_xlabel("Energy (Sum of Amplitudes)")
axes[0].set_ylabel("Density")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Peak amplitude
axes[1].hist(
    df[df["NLOS"] == 0]["CIR_PEAK_FOCUS"],
    bins=50,
    alpha=0.7,
    label="LOS",
    color="#2ecc71",
    density=True,
)
axes[1].hist(
    df[df["NLOS"] == 1]["CIR_PEAK_FOCUS"],
    bins=50,
    alpha=0.7,
    label="NLOS",
    color="#e74c3c",
    density=True,
)
axes[1].set_title(
    f"Peak Amplitude in Region {FOCUS_START}-{FOCUS_END}", fontweight="bold"
)
axes[1].set_xlabel("Peak Amplitude")
axes[1].set_ylabel("Density")
axes[1].legend()
axes[1].grid(alpha=0.3)

# Energy vs Range
axes[2].scatter(
    df[df["NLOS"] == 0]["RANGE"],
    df[df["NLOS"] == 0]["CIR_ENERGY_FOCUS"],
    c="#2ecc71",
    alpha=0.5,
    s=10,
    label="LOS",
)
axes[2].scatter(
    df[df["NLOS"] == 1]["RANGE"],
    df[df["NLOS"] == 1]["CIR_ENERGY_FOCUS"],
    c="#e74c3c",
    alpha=0.5,
    s=10,
    label="NLOS",
)
axes[2].set_title("CIR Energy vs Range", fontweight="bold")
axes[2].set_xlabel("Range (m)")
axes[2].set_ylabel("CIR Energy")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.suptitle("CIR Energy Analysis in Focused Region", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cir_energy_analysis.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: cir_energy_analysis.png")
plt.close()

# =============================================================================
# 5. MULTI-SAMPLE OVERLAY
# =============================================================================
print("Creating multi-sample overlay plot...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# LOS overlay
for i in range(10):
    sample = los_df.sample(1, random_state=i).iloc[0]
    ax1.plot(
        range(FOCUS_START, FOCUS_END),
        sample[FOCUS_COLUMNS].values,
        alpha=0.4,
        color="#2ecc71",
        linewidth=1,
    )
los_mean = los_df[FOCUS_COLUMNS].mean()
ax1.plot(
    range(FOCUS_START, FOCUS_END),
    los_mean.values,
    color="darkgreen",
    linewidth=3,
    label="Mean",
)
ax1.set_title(
    "LOS: 10 Random Samples + Mean (Focused View)", fontweight="bold", fontsize=12
)
ax1.set_xlabel("CIR Sample Index")
ax1.set_ylabel("Amplitude")
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_xticks(range(FOCUS_START, FOCUS_END + 1, 20))

# NLOS overlay
for i in range(10):
    sample = nlos_df.sample(1, random_state=i).iloc[0]
    ax2.plot(
        range(FOCUS_START, FOCUS_END),
        sample[FOCUS_COLUMNS].values,
        alpha=0.4,
        color="#e74c3c",
        linewidth=1,
    )
nlos_mean = nlos_df[FOCUS_COLUMNS].mean()
ax2.plot(
    range(FOCUS_START, FOCUS_END),
    nlos_mean.values,
    color="darkred",
    linewidth=3,
    label="Mean",
)
ax2.set_title(
    "NLOS: 10 Random Samples + Mean (Focused View)", fontweight="bold", fontsize=12
)
ax2.set_xlabel("CIR Sample Index")
ax2.set_ylabel("Amplitude")
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_xticks(range(FOCUS_START, FOCUS_END + 1, 20))

plt.suptitle("Multi-Sample Overlay Comparison", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "multi_sample_overlay.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: multi_sample_overlay.png")
plt.close()

# =============================================================================
# 6. STATISTICAL SUMMARY
# =============================================================================
print("=" * 80)
print("STATISTICAL SUMMARY - FOCUSED REGION ANALYSIS")
print("=" * 80)

print("\n📊 FP_IDX Statistics:")
print(
    f"  LOS  - Mean: {los_df['FP_IDX'].mean():.2f}, Std: {los_df['FP_IDX'].std():.2f}"
)
print(
    f"  NLOS - Mean: {nlos_df['FP_IDX'].mean():.2f}, Std: {nlos_df['FP_IDX'].std():.2f}"
)

print("\n📊 CIR Energy in Focused Region (730-850):")
print(
    f"  LOS  - Mean: {df[df['NLOS'] == 0]['CIR_ENERGY_FOCUS'].mean():,.0f}, Std: {df[df['NLOS'] == 0]['CIR_ENERGY_FOCUS'].std():,.0f}"
)
print(
    f"  NLOS - Mean: {df[df['NLOS'] == 1]['CIR_ENERGY_FOCUS'].mean():,.0f}, Std: {df[df['NLOS'] == 1]['CIR_ENERGY_FOCUS'].std():,.0f}"
)

print("\n📊 Peak Amplitude in Focused Region:")
print(
    f"  LOS  - Mean: {df[df['NLOS'] == 0]['CIR_PEAK_FOCUS'].mean():.0f}, Std: {df[df['NLOS'] == 0]['CIR_PEAK_FOCUS'].std():.0f}"
)
print(
    f"  NLOS - Mean: {df[df['NLOS'] == 1]['CIR_PEAK_FOCUS'].mean():.0f}, Std: {df[df['NLOS'] == 1]['CIR_PEAK_FOCUS'].std():.0f}"
)

print("\n📊 Feature Correlations with NLOS:")
new_features = ["CIR_ENERGY_FOCUS", "CIR_PEAK_FOCUS"]
for feature in new_features:
    corr = df[feature].corr(df["NLOS"])
    print(f"  {feature:20s}: {corr:+.3f}")

print("\n" + "=" * 80)
print("FOCUSED EDA COMPLETE!")
print("=" * 80)
print(f"\n💾 All focused visualizations saved to: {OUTPUT_DIR.absolute()}")

# List generated files
print("\nGenerated files:")
for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"  • {file.name}")

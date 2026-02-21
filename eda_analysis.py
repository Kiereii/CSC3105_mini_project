"""
UWB LOS/NLOS Dataset - Exploratory Data Analysis (EDA)
This script performs comprehensive EDA on the UWB dataset to understand:
- Dataset structure and basic statistics
- Class distribution (LOS vs NLOS)
- Feature distributions and relationships
- CIR (Channel Impulse Response) patterns
- Data quality assessment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Set style for better plots
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Configuration
DATA_DIR = Path("./Dataset/UWB-LOS-NLOS-Data-Set/dataset/Cleaned")
OUTPUT_DIR = Path("./eda_output")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("UWB LOS/NLOS DATASET - EXPLORATORY DATA ANALYSIS")
print("=" * 80)
print()

# =============================================================================
# 1. LOAD AND COMBINE DATA
# =============================================================================
print("Step 1: Loading and combining dataset...")
print("-" * 80)

csv_files = sorted(DATA_DIR.glob("uwb_cleaned_dataset_part*.csv"))
print(f"Found {len(csv_files)} CSV files")

dataframes = []
for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)
    print(f"  ✓ Loaded {file.name}: {df.shape[0]} rows, {df.shape[1]} columns")

# Combine all dataframes
df = pd.concat(dataframes, ignore_index=True)
print(f"\n✓ Combined dataset shape: {df.shape}")
print()

# =============================================================================
# 2. BASIC DATASET STATISTICS
# =============================================================================
print("Step 2: Basic Dataset Statistics")
print("-" * 80)

print(f"Total samples: {len(df):,}")
print(f"Total features: {df.shape[1]}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print()

# Identify feature types
cir_columns = [col for col in df.columns if col.startswith("CIR")]
non_cir_columns = [col for col in df.columns if not col.startswith("CIR")]

print(f"Non-CIR features: {len(non_cir_columns)}")
print(f"  - {non_cir_columns}")
print(f"CIR samples: {len(cir_columns)} (time-domain samples)")
print()

# Check data types
print("Data Types:")
print(df.dtypes.value_counts())
print()

# =============================================================================
# 3. CLASS DISTRIBUTION
# =============================================================================
print("Step 3: Class Distribution Analysis")
print("-" * 80)

# NLOS column: 1 = NLOS, 0 = LOS
class_counts = df["NLOS"].value_counts().sort_index()
class_percentages = df["NLOS"].value_counts(normalize=True).sort_index() * 100

print("Class Distribution:")
print(f"  LOS (0):  {class_counts[0]:,} samples ({class_percentages[0]:.2f}%)")
print(f"  NLOS (1): {class_counts[1]:,} samples ({class_percentages[1]:.2f}%)")
print(
    f"  Balance: {'✓ Balanced' if abs(class_percentages[0] - 50) < 5 else '⚠ Imbalanced'}"
)
print()

# Visualize class distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Bar plot
class_counts.plot(kind="bar", ax=ax1, color=["#2ecc71", "#e74c3c"])
ax1.set_title("Class Distribution (Count)", fontsize=14, fontweight="bold")
ax1.set_xlabel("Class")
ax1.set_ylabel("Count")
ax1.set_xticklabels(["LOS (0)", "NLOS (1)"], rotation=0)
ax1.grid(axis="y", alpha=0.3)

# Add count labels on bars
for i, v in enumerate(class_counts.values):
    ax1.text(
        i,
        v + 200,
        f"{v:,}\n({class_percentages.iloc[i]:.1f}%)",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

# Pie chart
ax2.pie(
    class_counts,
    labels=["LOS", "NLOS"],
    autopct="%1.1f%%",
    colors=["#2ecc71", "#e74c3c"],
    startangle=90,
)
ax2.set_title("Class Distribution (Percentage)", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: {OUTPUT_DIR / 'class_distribution.png'}")
plt.close()

# =============================================================================
# 4. MISSING VALUES & DATA QUALITY
# =============================================================================
print("Step 4: Data Quality Assessment")
print("-" * 80)

# Check for missing values
missing_values = df.isnull().sum()
missing_features = missing_values[missing_values > 0]

if len(missing_features) == 0:
    print("✓ No missing values found in the dataset")
else:
    print(f"⚠ Found {len(missing_features)} features with missing values:")
    print(missing_features[missing_features > 0])

print()

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates:,} ({duplicates / len(df) * 100:.3f}%)")
print()

# Basic statistics for non-CIR features
print("Descriptive Statistics (Non-CIR Features):")
non_cir_stats = df[non_cir_columns].describe()
print(non_cir_stats.round(3))
print()

# =============================================================================
# 5. FEATURE DISTRIBUTIONS
# =============================================================================
print("Step 5: Feature Distribution Analysis")
print("-" * 80)

# Key features to analyze (excluding CIR samples)
key_features = [
    "RANGE",
    "FP_IDX",
    "FP_AMP1",
    "FP_AMP2",
    "FP_AMP3",
    "STDEV_NOISE",
    "CIR_PWR",
    "MAX_NOISE",
    "RXPACC",
    "SNR",
    "SNR_dB",
]

# Create distribution plots
n_features = len(key_features)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

for idx, feature in enumerate(key_features):
    if feature in df.columns:
        ax = axes[idx]

        # Plot distributions for both classes
        los_data = df[df["NLOS"] == 0][feature]
        nlos_data = df[df["NLOS"] == 1][feature]

        ax.hist(
            los_data, bins=50, alpha=0.6, label="LOS", color="#2ecc71", density=True
        )
        ax.hist(
            nlos_data, bins=50, alpha=0.6, label="NLOS", color="#e74c3c", density=True
        )

        ax.set_title(f"{feature} Distribution", fontweight="bold")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.3)

# Hide empty subplots
for idx in range(n_features, len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_distributions.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: {OUTPUT_DIR / 'feature_distributions.png'}")
plt.close()

print(f"✓ Analyzed {n_features} key features")
print()

# =============================================================================
# 6. STATISTICAL COMPARISON BY CLASS
# =============================================================================
print("Step 6: Statistical Comparison (LOS vs NLOS)")
print("-" * 80)

comparison_stats = []
for feature in key_features:
    if feature in df.columns:
        los_mean = df[df["NLOS"] == 0][feature].mean()
        nlos_mean = df[df["NLOS"] == 1][feature].mean()
        los_std = df[df["NLOS"] == 0][feature].std()
        nlos_std = df[df["NLOS"] == 1][feature].std()

        comparison_stats.append(
            {
                "Feature": feature,
                "LOS_Mean": los_mean,
                "NLOS_Mean": nlos_mean,
                "Difference": nlos_mean - los_mean,
                "LOS_Std": los_std,
                "NLOS_Std": nlos_std,
            }
        )

comparison_df = pd.DataFrame(comparison_stats)
print(comparison_df.round(3).to_string(index=False))
print()

# =============================================================================
# 7. CORRELATION ANALYSIS
# =============================================================================
print("Step 7: Feature Correlation Analysis")
print("-" * 80)

# Correlation matrix for non-CIR features
corr_features = [
    "NLOS",
    "RANGE",
    "FP_IDX",
    "FP_AMP1",
    "FP_AMP2",
    "FP_AMP3",
    "STDEV_NOISE",
    "CIR_PWR",
    "MAX_NOISE",
    "RXPACC",
    "SNR",
    "SNR_dB",
]
corr_features = [f for f in corr_features if f in df.columns]

corr_matrix = df[corr_features].corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
plt.title("Feature Correlation Matrix", fontsize=16, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: {OUTPUT_DIR / 'correlation_matrix.png'}")
plt.close()

# Show correlations with target (NLOS)
print("\nCorrelations with NLOS (target):")
nlos_corr = corr_matrix["NLOS"].drop("NLOS").sort_values(key=abs, ascending=False)
for feature, corr in nlos_corr.items():
    direction = "↑" if corr > 0 else "↓"
    strength = (
        "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
    )
    print(f"  {feature:15s}: {corr:7.3f} {direction} ({strength})")
print()

# =============================================================================
# 8. CIR (CHANNEL IMPULSE RESPONSE) ANALYSIS
# =============================================================================
print("Step 8: CIR Pattern Analysis")
print("-" * 80)

# Sample a few CIR signals from each class
np.random.seed(42)
n_samples = 3

los_samples = df[df["NLOS"] == 0].sample(n_samples, random_state=42)
nlos_samples = df[df["NLOS"] == 1].sample(n_samples, random_state=42)

# Plot individual CIR samples
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for idx in range(n_samples):
    # LOS samples
    ax = axes[0, idx]
    cir_data = los_samples.iloc[idx][cir_columns].values
    ax.plot(cir_data, color="#2ecc71", linewidth=1.5)
    ax.set_title(f"LOS Sample {idx + 1}", fontweight="bold")
    ax.set_xlabel("CIR Sample Index")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)

    # NLOS samples
    ax = axes[1, idx]
    cir_data = nlos_samples.iloc[idx][cir_columns].values
    ax.plot(cir_data, color="#e74c3c", linewidth=1.5)
    ax.set_title(f"NLOS Sample {idx + 1}", fontweight="bold")
    ax.set_xlabel("CIR Sample Index")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)

plt.suptitle(
    "Individual CIR Samples Comparison", fontsize=16, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cir_samples_comparison.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: {OUTPUT_DIR / 'cir_samples_comparison.png'}")
plt.close()

# Plot mean CIR profiles
print("  Computing mean CIR profiles...")
los_mean_cir = df[df["NLOS"] == 0][cir_columns].mean()
nlos_mean_cir = df[df["NLOS"] == 1][cir_columns].mean()

plt.figure(figsize=(14, 6))
plt.plot(los_mean_cir.values, label="LOS (Mean)", color="#2ecc71", linewidth=2)
plt.plot(nlos_mean_cir.values, label="NLOS (Mean)", color="#e74c3c", linewidth=2)
plt.fill_between(
    range(len(los_mean_cir)), los_mean_cir.values, alpha=0.3, color="#2ecc71"
)
plt.fill_between(
    range(len(nlos_mean_cir)), nlos_mean_cir.values, alpha=0.3, color="#e74c3c"
)
plt.title("Mean CIR Profiles: LOS vs NLOS", fontsize=16, fontweight="bold")
plt.xlabel("CIR Sample Index")
plt.ylabel("Amplitude")
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mean_cir_profiles.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: {OUTPUT_DIR / 'mean_cir_profiles.png'}")
plt.close()

# =============================================================================
# 9. BOXPLOT COMPARISON
# =============================================================================
print("Step 9: Boxplot Comparison for Key Features")
print("-" * 80)

# Select most important features based on correlation
important_features = nlos_corr.head(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(important_features):
    ax = axes[idx]

    data_to_plot = [df[df["NLOS"] == 0][feature], df[df["NLOS"] == 1][feature]]
    bp = ax.boxplot(data_to_plot, labels=["LOS", "NLOS"], patch_artist=True)

    bp["boxes"][0].set_facecolor("#2ecc71")
    bp["boxes"][1].set_facecolor("#e74c3c")

    ax.set_title(f"{feature}", fontweight="bold")
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.3)

plt.suptitle(
    "Feature Distribution Comparison (Boxplots)", fontsize=16, fontweight="bold", y=1.00
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_boxplots.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: {OUTPUT_DIR / 'feature_boxplots.png'}")
plt.close()

# =============================================================================
# 10. SUMMARY AND KEY INSIGHTS
# =============================================================================
print("=" * 80)
print("EDA SUMMARY & KEY INSIGHTS")
print("=" * 80)

print(f"""
📊 DATASET OVERVIEW:
   • Total samples: {len(df):,} ({class_counts[0]:,} LOS, {class_counts[1]:,} NLOS)
   • Features: {len(non_cir_columns)} non-CIR + {len(cir_columns)} CIR samples
   • Balance: {"Perfectly balanced ✓" if abs(class_percentages[0] - 50) < 1 else "Slightly imbalanced"}
   • Quality: {"No missing values ✓" if len(missing_features) == 0 else f"{len(missing_features)} features with missing values"}

📈 KEY FINDINGS:
""")

# Top discriminative features
print("   Most Discriminative Features (by correlation with NLOS):")
for i, (feature, corr) in enumerate(nlos_corr.head(5).items(), 1):
    print(f"   {i}. {feature}: {corr:+.3f}")

print(f"""
🎯 RECOMMENDATIONS FOR NEXT STEPS:
   1. Use top {len(nlos_corr.head(5))} correlated features as primary predictors
   2. Consider CIR pattern analysis for deep learning approaches
   3. No need for class balancing (dataset is well-balanced)
   4. Normalize features before model training (different scales observed)
   5. Investigate feature engineering from CIR data (e.g., peak detection)

💾 OUTPUT FILES:
   All visualizations saved to: {OUTPUT_DIR.absolute()}
""")

print("=" * 80)
print("EDA COMPLETE!")
print("=" * 80)

# Save summary statistics to file
summary_file = OUTPUT_DIR / "eda_summary.txt"
with open(summary_file, "w") as f:
    f.write("UWB LOS/NLOS Dataset - EDA Summary\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Dataset Shape: {df.shape}\n")
    f.write(f"Class Distribution:\n")
    f.write(f"  LOS:  {class_counts[0]:,} ({class_percentages[0]:.2f}%)\n")
    f.write(f"  NLOS: {class_counts[1]:,} ({class_percentages[1]:.2f}%)\n\n")
    f.write("Feature Correlations with NLOS:\n")
    for feature, corr in nlos_corr.items():
        f.write(f"  {feature}: {corr:.3f}\n")

print(f"✓ Summary saved to: {summary_file}")

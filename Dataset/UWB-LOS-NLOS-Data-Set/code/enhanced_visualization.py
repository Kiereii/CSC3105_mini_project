"""
enhanced_visualization.py - EDA, Feature Analysis & PCA
=========================================================
Produces all visualizations required for the mini project report:

  a.I   - PCA Data Reduction (CIR dimensionality)
  a.II  - Feature Extraction (SNR comparison)
  a.III - Class Balance
  a.IV  - Outlier Detection
  a.V   - Feature Importance Ranking + Correlation Heatmap
  a.VI  - Signal Sample Plots (LOS vs NLOS CIR fingerprints)

Imports shared loader and column definitions from uwb_dataset.py.
Run with DATA_TYPE = 'Cleaned' for report outputs.
Run with DATA_TYPE = 'Raw'     to compare before/after cleaning.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Import shared utilities from uwb_dataset.py ──────────────────────────────
from uwb_dataset import (
    load_data,
    get_cir_columns,
    get_feature_columns,
    split_by_class,
    CIR_COLUMNS,
)

# =============================================================================
# CONFIGURATION  ←  only change things here
# =============================================================================
DATA_TYPE       = 'Cleaned'   # 'Cleaned' or 'Raw'
N_SIGNAL_PLOTS  = 30          # how many sample CIR traces to overlay per plot
CIR_ZOOM_START  = 730         # x-axis zoom start for signal plots
CIR_ZOOM_END    = 850         # x-axis zoom end  for signal plots
PCA_VARIANCE_TARGET = 0.95    # target explained variance for PCA cutoff
RANDOM_SEED     = 42
# =============================================================================

# ── Global plot style ─────────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'figure.dpi': 100,
})

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
df = load_data(DATA_TYPE, verbose=True)

cir_cols  = get_cir_columns(df)           # actual CIR columns present
feat_cols = get_feature_columns(df, include_engineered=True)
los_df, nlos_df = split_by_class(df)

print(f"\nCIR columns present : {len(cir_cols)}")
print(f"Feature columns     : {feat_cols}")


# =============================================================================
# STEP 2: DATA QUALITY AUDIT  (printed, not plotted)
# =============================================================================
print("\n─── Data Quality Audit ───────────────────────────────────")
print(f"  Total rows         : {len(df):,}")
print(f"  Total null values  : {df.isnull().sum().sum()}")
print(f"  Duplicate rows     : {df.duplicated().sum()}")
if 'RANGE' in df.columns:
    print(f"  Negative RANGE rows: {(df['RANGE'] <= 0).sum()}")
print("──────────────────────────────────────────────────────────\n")


# =============================================================================
# STEP 3: MAIN DASHBOARD  (3×2 grid, Req a.III / a.IV / a.VI)
# =============================================================================
fig, axes = plt.subplots(3, 2, figsize=(20, 18))
fig.suptitle(
    f"UWB LOS/NLOS Dataset — Exploratory Analysis ({DATA_TYPE})",
    fontsize=16, fontweight='bold', y=1.01
)

# ── Plot 1: LOS CIR signal samples ───────────────────────────────────────────
ax = axes[0, 0]
ax.set_title("LOS Signal Samples (CIR)")
sample_los = los_df.sample(min(N_SIGNAL_PLOTS, len(los_df)),
                           random_state=RANDOM_SEED)
for _, row in sample_los.iterrows():
    ax.plot(row[cir_cols].values, alpha=0.5, linewidth=0.8)
ax.set_xlim(CIR_ZOOM_START, CIR_ZOOM_END)
ax.set_xticks(np.arange(CIR_ZOOM_START, CIR_ZOOM_END + 1, 10))
ax.set_xlabel("CIR Tap Index")
ax.set_ylabel("Amplitude")

# ── Plot 2: NLOS CIR signal samples ──────────────────────────────────────────
ax = axes[0, 1]
ax.set_title("NLOS Signal Samples (CIR)")
sample_nlos = nlos_df.sample(min(N_SIGNAL_PLOTS, len(nlos_df)),
                              random_state=RANDOM_SEED)
for _, row in sample_nlos.iterrows():
    ax.plot(row[cir_cols].values, alpha=0.5, linewidth=0.8)
ax.set_xlim(CIR_ZOOM_START, CIR_ZOOM_END)
ax.set_xticks(np.arange(CIR_ZOOM_START, CIR_ZOOM_END + 1, 10))
ax.set_xlabel("CIR Tap Index")
ax.set_ylabel("Amplitude")

# ── Plot 3: Class balance pie  (Req a.III) ────────────────────────────────────
ax = axes[1, 0]
ax.set_title("Dataset Class Balance  (Req a.III)")
counts = [len(los_df), len(nlos_df)]
wedges, texts, autotexts = ax.pie(
    counts,
    labels=[f'LOS\n({counts[0]:,})', f'NLOS\n({counts[1]:,})'],
    autopct='%1.1f%%',
    colors=['#5B9BD5', '#ED7D31'],
    startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)
for at in autotexts:
    at.set_fontsize(11)

# ── Plot 4: Range distribution histogram ─────────────────────────────────────
ax = axes[1, 1]
ax.set_title("Measured Range Distribution")
if 'RANGE' in df.columns:
    ax.hist(los_df['RANGE'],  bins=40, alpha=0.6, label='LOS',  color='#5B9BD5')
    ax.hist(nlos_df['RANGE'], bins=40, alpha=0.6, label='NLOS', color='#ED7D31')
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Count")
    ax.legend()
else:
    ax.text(0.5, 0.5, "RANGE column not found", ha='center', va='center')

# ── Plot 5: Mean CIR profiles (LOS vs NLOS fingerprint) ──────────────────────
ax = axes[2, 0]
ax.set_title("Mean CIR Profile — Signal Fingerprint")
ax.plot(df[df['NLOS'] == 0][cir_cols].mean().values,
        label='Mean LOS',  color='#5B9BD5', linewidth=2)
ax.plot(df[df['NLOS'] == 1][cir_cols].mean().values,
        label='Mean NLOS', color='#ED7D31', linewidth=2)
ax.set_xlabel("CIR Tap Index")
ax.set_ylabel("Mean Amplitude")
ax.legend()

# ── Plot 6: Range box-plot outlier detection  (Req a.IV) ─────────────────────
ax = axes[2, 1]
ax.set_title("Range Outlier Detection  (Req a.IV)")
if 'RANGE' in df.columns:
    bp = ax.boxplot(
        [los_df['RANGE'].dropna(), nlos_df['RANGE'].dropna()],
        labels=['LOS', 'NLOS'],
        patch_artist=True,
        medianprops={'color': 'black', 'linewidth': 2}
    )
    bp['boxes'][0].set_facecolor('#5B9BD5')
    bp['boxes'][1].set_facecolor('#ED7D31')
    ax.set_ylabel("Range (m)")
else:
    ax.text(0.5, 0.5, "RANGE column not found", ha='center', va='center')

plt.tight_layout()
plt.savefig("dashboard_main.png", bbox_inches='tight', dpi=150)
print("Saved: dashboard_main.png")
plt.show()


# =============================================================================
# STEP 4: SNR FEATURE EXTRACTION PLOT  (Req a.II)
# Only shown when running on Cleaned data (SNR_dB added by clean_local.py)
# =============================================================================
if 'SNR_dB' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Feature Extraction: SNR Analysis  (Req a.II)",
                 fontsize=14, fontweight='bold')

    # Box plot comparison
    sns.boxplot(x='NLOS', y='SNR_dB', data=df,
                palette=['#5B9BD5', '#ED7D31'], ax=axes[0])
    axes[0].set_title("SNR_dB — LOS vs NLOS")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['LOS', 'NLOS'])
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("SNR (dB)")

    # Distribution overlay
    axes[1].set_title("SNR_dB Distribution")
    los_df['SNR_dB'].plot(kind='hist',  bins=50, alpha=0.6,
                          color='#5B9BD5', label='LOS',  ax=axes[1])
    nlos_df['SNR_dB'].plot(kind='hist', bins=50, alpha=0.6,
                           color='#ED7D31', label='NLOS', ax=axes[1])
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("snr_feature_analysis.png", bbox_inches='tight', dpi=150)
    print("Saved: snr_feature_analysis.png")
    plt.show()
else:
    print("[INFO] SNR_dB column not found — skipping SNR plot (run on Cleaned data).")


# =============================================================================
# STEP 5: FEATURE IMPORTANCE RANKING + CORRELATION HEATMAP  (Req a.V)
# =============================================================================
# Build list of available non-CIR feature columns including NLOS label
heatmap_features = ['NLOS'] + [c for c in feat_cols if c != 'NLOS']

if len(heatmap_features) >= 2:
    corr_matrix = df[heatmap_features].corr()
    ranking = (
        corr_matrix['NLOS']
        .abs()
        .sort_values(ascending=False)
        .round(4)
        .drop('NLOS')           # remove self-correlation
    )

    print("\n─── Feature Importance Ranking (|correlation with NLOS|) ────")
    for feat, val in ranking.items():
        bar = '█' * int(val * 30)
        print(f"  {feat:<15} {val:.4f}  {bar}")
    print("──────────────────────────────────────────────────────────────\n")

    # Heatmap
    plt.figure(figsize=(10, 8))
    mask = np.zeros_like(corr_matrix, dtype=bool)   # show full matrix
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=0.5,
        vmin=-1, vmax=1,
        annot_kws={"size": 10}
    )
    plt.title(f"Feature Correlation Heatmap  (Req a.V) — {DATA_TYPE} Data",
              fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("feature_correlation_heatmap.png", bbox_inches='tight', dpi=150)
    print("Saved: feature_correlation_heatmap.png")
    plt.show()
else:
    print("[INFO] Not enough feature columns for heatmap.")


# =============================================================================
# STEP 6: PCA — CUMULATIVE EXPLAINED VARIANCE  (Req a.I)
# =============================================================================
print("\n─── PCA on CIR Features ──────────────────────────────────────")
print(f"  Using {len(cir_cols)} CIR columns")
print("  Step 1: Standardising data...")

x_cir = df[cir_cols].values

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_cir)

print("  Step 2: Fitting PCA (this may take ~1–2 minutes on full dataset)...")
pca = PCA(random_state=RANDOM_SEED)
pca.fit(x_scaled)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_total = len(cir_cols)

# Find cut-off components for each threshold
components_90 = int(np.argmax(cumulative_variance >= 0.90)) + 1
components_95 = int(np.argmax(cumulative_variance >= PCA_VARIANCE_TARGET)) + 1
components_99 = int(np.argmax(cumulative_variance >= 0.99)) + 1

print(f"\n  Original feature count  : {n_total}")
print(f"  Components for 90% var  : {components_90}  "
      f"(reduction: {((n_total - components_90)/n_total*100):.1f}%)")
print(f"  Components for 95% var  : {components_95}  "
      f"(reduction: {((n_total - components_95)/n_total*100):.1f}%)")
print(f"  Components for 99% var  : {components_99}  "
      f"(reduction: {((n_total - components_99)/n_total*100):.1f}%)")
print("──────────────────────────────────────────────────────────────\n")

# ── PCA plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle("PCA: CIR Data Reduction  (Req a.I)",
             fontsize=14, fontweight='bold')

# Left: cumulative explained variance curve
ax = axes[0]
ax.plot(range(1, n_total + 1), cumulative_variance, color='royalblue', lw=2)
ax.axhline(y=0.90, color='orange',  linestyle='--', linewidth=1.2, label='90% variance')
ax.axhline(y=PCA_VARIANCE_TARGET, color='red', linestyle='--',
           linewidth=1.2, label=f'{int(PCA_VARIANCE_TARGET*100)}% variance')
ax.axhline(y=0.99, color='purple', linestyle='--', linewidth=1.2, label='99% variance')
ax.axvline(x=components_95, color='green', linestyle='--', linewidth=1.5,
           label=f'{components_95} components @ 95%')
ax.set_title("Cumulative Explained Variance")
ax.set_xlabel("Number of Principal Components")
ax.set_ylabel("Cumulative Explained Variance")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.4)

# Right: individual explained variance (scree plot) — first 100 components
ax = axes[1]
individual_var = pca.explained_variance_ratio_[:100]
ax.bar(range(1, 101), individual_var * 100, color='royalblue', alpha=0.75)
ax.set_title("Scree Plot — Top 100 Components")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Individual Explained Variance (%)")
ax.grid(True, alpha=0.4, axis='y')

plt.tight_layout()
plt.savefig("pca_analysis.png", bbox_inches='tight', dpi=150)
print("Saved: pca_analysis.png")
plt.show()

print("\n[DONE] All visualizations complete.")
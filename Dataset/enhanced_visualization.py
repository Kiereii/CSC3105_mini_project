import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import glob
import os
import sys
from pathlib import Path

# --- 1. FINAL CORRECTED PATH RESOLUTION ---
# TOGGLE THIS: Change to 'Raw' to see raw data, or 'Cleaned' to see your cleaned results
DATA_TYPE = 'Raw'  

script_dir = Path(__file__).resolve().parent

# Logic: Start where the script is (CSC3105_mini_project)
# Navigates the specific nested folders shown in your sidebar
data_dir = script_dir / 'Dataset' / 'UWB-LOS-NLOS-Data-Set' / 'dataset' / DATA_TYPE

# Fallback: In case the script is already inside the 'Dataset' or 'code' folder
if not data_dir.exists():
    data_dir = script_dir.parent / 'Dataset' / 'UWB-LOS-NLOS-Data-Set' / 'dataset' / DATA_TYPE

# Search for files based on the toggle
pattern = 'uwb_cleaned_dataset_part*.csv' if DATA_TYPE == 'Cleaned' else 'uwb_dataset_part*.csv'
csv_paths = sorted(glob.glob(str(data_dir / pattern)))

if not csv_paths:
    raise FileNotFoundError(
        f"Could not find {DATA_TYPE} files at: {data_dir.absolute()}\n"
        f"Please ensure folder names match your sidebar: 'Dataset' (Cap) and 'dataset' (small)."
    )

print(f"--- Success! Running Visualization on {DATA_TYPE} Data ---")
df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
print(f"Total samples loaded: {len(df)}")

# --- 2. DATA SEGREGATION ---
# Extract the 1016 CIR samples for fingerprint analysis
cir_columns = [f'CIR{i}' for i in range(1016)]
los_samples = df[df['NLOS'] == 0]
nlos_samples = df[df['NLOS'] == 1]

# --- 3. CORE VISUALIZATIONS (Requirement a.VI) ---
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(3, 2, figsize=(20, 18))

# 1 & 2: Sample Signal Plots (Integrity Check)
axes[0, 0].set_title("Line-of-Sight (LOS) Signal Samples", fontweight='bold')
for i in range(min(3, len(los_samples))):
    axes[0, 0].plot(los_samples.sample(1)[cir_columns].values.flatten(), alpha=0.6)

axes[0, 1].set_title("Non-Line-of-Sight (NLOS) Signal Samples", fontweight='bold')
for i in range(min(3, len(nlos_samples))):
    axes[0, 1].plot(nlos_samples.sample(1)[cir_columns].values.flatten(), alpha=0.6)

# 3. Class Balance (Requirement a.III)
axes[1, 0].set_title("Dataset Class Balance", fontweight='bold')
axes[1, 0].pie([len(los_samples), len(nlos_samples)], labels=['LOS', 'NLOS'], autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])

# 4. Measured Range Distribution
axes[1, 1].set_title("Measured Range Distribution", fontweight='bold')
axes[1, 1].hist(los_samples['RANGE'], bins=30, alpha=0.5, label='LOS', color='blue')
axes[1, 1].hist(nlos_samples['RANGE'], bins=30, alpha=0.5, label='NLOS', color='red')
axes[1, 1].legend()

# 5. Average Signal Profiles (Theoretical Evidence)
axes[2, 0].set_title("Average CIR Profile (Signal Fingerprint)", fontweight='bold')
axes[2, 0].plot(df[df['NLOS']==0][cir_columns].mean(), label='Mean LOS', color='blue', linewidth=2)
axes[2, 0].plot(df[df['NLOS']==1][cir_columns].mean(), label='Mean NLOS', color='red', linewidth=2)
axes[2, 0].legend()

# 6. Outlier Detection (Requirement a.IV)
axes[2, 1].set_title("Range Outlier Detection", fontweight='bold')
axes[2, 1].boxplot([los_samples['RANGE'], nlos_samples['RANGE']], labels=['LOS', 'NLOS'])

plt.tight_layout()
plt.show()

# --- 4. CONDITIONAL FEATURE VISUALS (Requirement a.II) ---
# Visualizes the new SNR features only if running on 'Cleaned' data
if 'SNR_dB' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='NLOS', y='SNR_dB', data=df, palette='Set2')
    plt.title("Feature Extraction: SNR Comparison (LOS vs NLOS)", fontweight='bold')
    plt.xticks([0, 1], ['LOS', 'NLOS'])
    plt.show()

# --- 5. FEATURE IMPORTANCE RANKING (Requirement a.V) ---
potential_features = ['NLOS', 'RANGE', 'FP_AMP1', 'FP_AMP2', 'STDEV_NOISE', 'CIR_PWR', 'SNR_dB']
existing_features = [f for f in potential_features if f in df.columns]

if 'NLOS' in df.columns:
    corr_matrix = df[existing_features].corr()
    ranking = corr_matrix['NLOS'].abs().sort_values(ascending=False).round(4)
    
    print(f"\n--- Feature Importance Ranking ({DATA_TYPE} Data) ---")
    print(ranking)

    # Creating the Heatmap in its own separate window
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f"Feature Correlation Heatmap ({DATA_TYPE} Data)", fontsize=14, fontweight='bold')

# --- FINAL STEP: DISPLAY ALL WINDOWS AT ONCE ---
# By putting this only at the very bottom, all charts (Main, SNR, and Heatmap) 
# will pop up simultaneously as separate windows.
plt.show() 

# --- 6. DATA QUALITY AUDIT ---
null_count = df.isnull().sum().sum()
print(f"\nTotal Null Values in {DATA_TYPE} set: {null_count}")
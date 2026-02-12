import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the dataset (all parts)
import glob, os
data_dir = '/home/keyreii/Documents/Data Analytics Mini Project/Dataset/UWB-LOS-NLOS-Data-Set/dataset'
csv_paths = sorted(glob.glob(os.path.join(data_dir, 'uwb_dataset_part*.csv')))
if not csv_paths:
    raise FileNotFoundError(f'No CSV files found in {data_dir}')
df_list = [pd.read_csv(p) for p in csv_paths]
df = pd.concat(df_list, ignore_index=True)
print(f"Loaded {len(csv_paths)} files, total rows: {len(df)}")

# Extract the CIR columns (CIR0 to CIR1015)
cir_columns = [f'CIR{i}' for i in range(1016)]
cir_data = df[cir_columns]

# Separate into LOS (Class 0) and NLOS (Class 1)
los_samples = df[df['NLOS'] == 0]
nlos_samples = df[df['NLOS'] == 1]

print(f"Total samples: {len(df)}")
print(f"LOS samples: {len(los_samples)}")
print(f"NLOS samples: {len(nlos_samples)}")

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# Create comprehensive visualizations
fig, axes = plt.subplots(3, 2, figsize=(20, 18))

# 1. Plot 3 Random LOS Samples
axes[0, 0].set_title("Line-of-Sight (LOS) Signals", fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel("Time (ns)", fontsize=12)
axes[0, 0].set_ylabel("Amplitude", fontsize=12)
for i in range(min(3, len(los_samples))):  # Ensure we don't exceed available samples
    sample = los_samples.sample(1)
    axes[0, 0].plot(sample[cir_columns].values.flatten(), alpha=0.7, label=f'LOS Sample {i+1}')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Plot 3 Random NLOS Samples
axes[0, 1].set_title("Non-Line-of-Sight (NLOS) Signals", fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel("Time (ns)", fontsize=12)
axes[0, 1].set_ylabel("Amplitude", fontsize=12)
for i in range(min(3, len(nlos_samples))):  # Ensure we don't exceed available samples
    sample = nlos_samples.sample(1)
    axes[0, 1].plot(sample[cir_columns].values.flatten(), alpha=0.7, label=f'NLOS Sample {i+1}')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Distribution of NLOS vs LOS
axes[1, 0].set_title("Distribution of LOS vs NLOS", fontsize=14, fontweight='bold')
axes[1, 0].pie([len(los_samples), len(nlos_samples)], labels=['LOS (0)', 'NLOS (1)'], autopct='%1.1f%%', startangle=90)
axes[1, 0].axis('equal')

# 4. Histogram of RANGE values for both classes
axes[1, 1].set_title("Range Distribution by Class", fontsize=14, fontweight='bold')
axes[1, 1].hist(los_samples['RANGE'], bins=30, alpha=0.5, label='LOS', color='skyblue', edgecolor='black')
axes[1, 1].hist(nlos_samples['RANGE'], bins=30, alpha=0.5, label='NLOS', color='lightcoral', edgecolor='black')
axes[1, 1].set_xlabel("Range", fontsize=12)
axes[1, 1].set_ylabel("Frequency", fontsize=12)
axes[1, 1].legend()

# 5. Mean CIR profile for LOS and NLOS
mean_cir_los = cir_data[df['NLOS'] == 0].mean(axis=0)
mean_cir_nlos = cir_data[df['NLOS'] == 1].mean(axis=0)

axes[2, 0].set_title("Mean CIR Profile - LOS vs NLOS", fontsize=14, fontweight='bold')
axes[2, 0].plot(mean_cir_los.values, label='LOS', color='blue', linewidth=2)
axes[2, 0].plot(mean_cir_nlos.values, label='NLOS', color='red', linewidth=2)
axes[2, 0].set_xlabel("Time (ns)", fontsize=12)
axes[2, 0].set_ylabel("Average Amplitude", fontsize=12)
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# 6. Box plot for RANGE by class (fixed)
axes[2, 1].boxplot([los_samples['RANGE'], nlos_samples['RANGE']], labels=['LOS', 'NLOS'])
axes[2, 1].set_title("Range Distribution by Class (Box Plot)", fontsize=14, fontweight='bold')
axes[2, 1].set_ylabel("Range", fontsize=12)
axes[2, 1].set_xlabel("Class", fontsize=12)

plt.tight_layout()
plt.show()

# Additional detailed visualizations
fig, axes = plt.subplots(2, 2, figsize=(20, 12))

# Correlation heatmap for key features
key_features = ['NLOS', 'RANGE', 'FP_AMP1', 'FP_AMP2', 'STDEV_NOISE', 'CIR_PWR']
corr_matrix = df[key_features].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 0], 
            cbar_kws={'shrink': 0.8}, square=True)
axes[0, 0].set_title("Correlation Heatmap of Key Features", fontsize=14, fontweight='bold')

# Scatter plot of RANGE vs First Peak Amplitude (FP_AMP1)
axes[0, 1].scatter(los_samples['RANGE'], los_samples['FP_AMP1'], alpha=0.5, label='LOS', color='blue', s=20)
axes[0, 1].scatter(nlos_samples['RANGE'], nlos_samples['FP_AMP1'], alpha=0.5, label='NLOS', color='red', s=20)
axes[0, 1].set_title("Range vs First Peak Amplitude", fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel("Range")
axes[0, 1].set_ylabel("First Peak Amplitude (FP_AMP1)")
axes[0, 1].legend()

# Distribution of signal power by class
axes[1, 0].hist(los_samples['CIR_PWR'], bins=30, alpha=0.5, label='LOS', color='skyblue', edgecolor='black')
axes[1, 0].hist(nlos_samples['CIR_PWR'], bins=30, alpha=0.5, label='NLOS', color='lightcoral', edgecolor='black')
axes[1, 0].set_title("CIR Power Distribution by Class", fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel("CIR Power")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].legend()

# Violin plot for RANGE by class
data_for_violin = [los_samples['RANGE'], nlos_samples['RANGE']]
axes[1, 1].violinplot(data_for_violin, positions=[1, 2], showmeans=True)
axes[1, 1].set_xticks([1, 2])
axes[1, 1].set_xticklabels(['LOS', 'NLOS'])
axes[1, 1].set_title("Range Distribution by Class (Violin Plot)", fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel("Range")

plt.tight_layout()
plt.show()

# Print some statistics
print("\nDetailed Statistics for LOS samples:")
print(los_samples[['NLOS', 'RANGE', 'FP_AMP1', 'FP_AMP2', 'STDEV_NOISE', 'CIR_PWR']].describe())

print("\nDetailed Statistics for NLOS samples:")
print(nlos_samples[['NLOS', 'RANGE', 'FP_AMP1', 'FP_AMP2', 'STDEV_NOISE', 'CIR_PWR']].describe())

# Calculate and display differences between classes
print("\nKey Differences Between LOS and NLOS:")
print(f"Average Range - LOS: {los_samples['RANGE'].mean():.2f}, NLOS: {nlos_samples['RANGE'].mean():.2f}")
print(f"Average FP_AMP1 - LOS: {los_samples['FP_AMP1'].mean():.2f}, NLOS: {nlos_samples['FP_AMP1'].mean():.2f}")
print(f"Average CIR_PWR - LOS: {los_samples['CIR_PWR'].mean():.2f}, NLOS: {nlos_samples['CIR_PWR'].mean():.2f}")
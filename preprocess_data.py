"""
UWB LOS/NLOS Dataset - Data Preprocessing Script
Option B: Core 15 features + CIR samples 730-850
Saves both StandardScaler and MinMaxScaler versions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import warnings

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = Path("./Dataset/UWB-LOS-NLOS-Data-Set/dataset/Cleaned")
OUTPUT_DIR = Path("./preprocessed_data")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
TEST_SIZE = 0.2  # 80/20 split

# Focused CIR region (from EDA)
CIR_START = 730
CIR_END = 850

print("=" * 80)
print("UWB DATASET PREPROCESSING - OPTION B")
print("=" * 80)
print(
    f"Features: 15 core + {CIR_END - CIR_START} CIR samples = {15 + (CIR_END - CIR_START)} total"
)
print(f"Train/Test Split: {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}")
print(f"Random Seed: {RANDOM_SEED}")
print("=" * 80)
print()

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("Step 1: Loading dataset...")

csv_files = sorted(DATA_DIR.glob("uwb_cleaned_dataset_part*.csv"))
print(f"Found {len(csv_files)} CSV files")

df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
print(f"✓ Loaded {len(df):,} samples")
print()

# =============================================================================
# STEP 2: FEATURE EXTRACTION
# =============================================================================
print("Step 2: Extracting features...")

# Core 15 features (exclude NLOS - that's our target)
core_features = [
    "RANGE",
    "FP_IDX",
    "FP_AMP1",
    "FP_AMP2",
    "FP_AMP3",
    "STDEV_NOISE",
    "CIR_PWR",
    "MAX_NOISE",
    "RXPACC",
    "CH",
    "FRAME_LEN",
    "PREAM_LEN",
    "BITRATE",
    "PRFR",
    "SNR",
    "SNR_dB",
]

# CIR samples 730-850
cir_features = [f"CIR{i}" for i in range(CIR_START, CIR_END)]

# Combine all feature names
feature_columns = core_features + cir_features
print(f"Core features: {len(core_features)}")
print(f"CIR features ({CIR_START}-{CIR_END}): {len(cir_features)}")
print(f"Total features: {len(feature_columns)}")
print()

# Extract features (X) and target (y)
X = df[feature_columns].values
y = df["NLOS"].values

print(f"✓ Feature matrix shape: {X.shape}")
print(f"✓ Target vector shape: {y.shape}")
print()

# =============================================================================
# STEP 3: TRAIN/TEST SPLIT
# =============================================================================
print("Step 3: Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y,  # Maintain LOS/NLOS balance
)

print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")
print()

# Verify balance
train_balance = np.bincount(y_train)
test_balance = np.bincount(y_test)
print("Class distribution:")
print(
    f"  Training - LOS: {train_balance[0]:,} ({train_balance[0] / len(y_train) * 100:.1f}%), NLOS: {train_balance[1]:,} ({train_balance[1] / len(y_train) * 100:.1f}%)"
)
print(
    f"  Test - LOS: {test_balance[0]:,} ({test_balance[0] / len(y_test) * 100:.1f}%), NLOS: {test_balance[1]:,} ({test_balance[1] / len(y_test) * 100:.1f}%)"
)
print()

# =============================================================================
# STEP 4: SAVE UNPROCESSED DATA (for algorithms that don't need scaling)
# =============================================================================
print("Step 4: Saving unprocessed data...")

np.save(OUTPUT_DIR / "X_train_unscaled.npy", X_train)
np.save(OUTPUT_DIR / "X_test_unscaled.npy", X_test)
np.save(OUTPUT_DIR / "y_train.npy", y_train)
np.save(OUTPUT_DIR / "y_test.npy", y_test)

print(f"✓ Saved: X_train_unscaled.npy ({X_train.shape})")
print(f"✓ Saved: X_test_unscaled.npy ({X_test.shape})")
print(f"✓ Saved: y_train.npy ({y_train.shape})")
print(f"✓ Saved: y_test.npy ({y_test.shape})")
print()

# =============================================================================
# STEP 5: STANDARD SCALER (mean=0, std=1)
# =============================================================================
print("Step 5: Applying StandardScaler...")

scaler_standard = StandardScaler()
X_train_standard = scaler_standard.fit_transform(X_train)
X_test_standard = scaler_standard.transform(X_test)

# Save scaled data
np.save(OUTPUT_DIR / "X_train_standard.npy", X_train_standard)
np.save(OUTPUT_DIR / "X_test_standard.npy", X_test_standard)

# Save scaler
with open(OUTPUT_DIR / "scaler_standard.pkl", "wb") as f:
    pickle.dump(scaler_standard, f)

print(f"✓ Saved: X_train_standard.npy")
print(f"✓ Saved: X_test_standard.npy")
print(f"✓ Saved: scaler_standard.pkl")
print()

# =============================================================================
# STEP 6: MINMAX SCALER (range 0-1)
# =============================================================================
print("Step 6: Applying MinMaxScaler...")

scaler_minmax = MinMaxScaler()
X_train_minmax = scaler_minmax.fit_transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)

# Save scaled data
np.save(OUTPUT_DIR / "X_train_minmax.npy", X_train_minmax)
np.save(OUTPUT_DIR / "X_test_minmax.npy", X_test_minmax)

# Save scaler
with open(OUTPUT_DIR / "scaler_minmax.pkl", "wb") as f:
    pickle.dump(scaler_minmax, f)

print(f"✓ Saved: X_train_minmax.npy")
print(f"✓ Saved: X_test_minmax.npy")
print(f"✓ Saved: scaler_minmax.pkl")
print()

# =============================================================================
# STEP 7: SAVE METADATA
# =============================================================================
print("Step 7: Saving metadata...")

# Save feature names
with open(OUTPUT_DIR / "feature_names.txt", "w") as f:
    f.write("FEATURE NAMES (Option B)\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Core Features ({len(core_features)}):\n")
    for i, feat in enumerate(core_features, 1):
        f.write(f"  {i:2d}. {feat}\n")
    f.write(f"\nCIR Features ({len(cir_features)}):\n")
    f.write(f"  Range: CIR{CIR_START} to CIR{CIR_END - 1}\n")
    f.write(f"  (First path region based on EDA)\n")

# Save preprocessing info
with open(OUTPUT_DIR / "preprocessing_info.txt", "w") as f:
    f.write("PREPROCESSING INFORMATION\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Original dataset size: {len(df):,} samples\n")
    f.write(
        f"Training set size: {len(X_train):,} samples ({(1 - TEST_SIZE) * 100:.0f}%)\n"
    )
    f.write(f"Test set size: {len(X_test):,} samples ({TEST_SIZE * 100:.0f}%)\n")
    f.write(f"Number of features: {len(feature_columns)}\n")
    f.write(f"  - Core features: {len(core_features)}\n")
    f.write(f"  - CIR features: {len(cir_features)} (samples {CIR_START}-{CIR_END})\n")
    f.write(f"\nRandom seed: {RANDOM_SEED}\n")
    f.write(f"Stratified split: Yes\n\n")
    f.write("Generated files:\n")
    f.write("  1. X_train_unscaled.npy / X_test_unscaled.npy\n")
    f.write("  2. X_train_standard.npy / X_test_standard.npy\n")
    f.write("  3. X_train_minmax.npy / X_test_minmax.npy\n")
    f.write("  4. y_train.npy / y_test.npy\n")
    f.write("  5. scaler_standard.pkl\n")
    f.write("  6. scaler_minmax.pkl\n")

print(f"✓ Saved: feature_names.txt")
print(f"✓ Saved: preprocessing_info.txt")
print()

# =============================================================================
# STEP 8: VERIFICATION
# =============================================================================
print("Step 8: Verification...")
print()

# Check StandardScaler
print("StandardScaler statistics (first 5 features):")
means = X_train_standard.mean(axis=0)[:5]
stds = X_train_standard.std(axis=0)[:5]
for i in range(5):
    print(f"  {feature_columns[i]:15s}: mean={means[i]:8.3f}, std={stds[i]:6.3f}")
print()

# Check MinMaxScaler
print("MinMaxScaler statistics (first 5 features):")
mins = X_train_minmax.min(axis=0)[:5]
maxs = X_train_minmax.max(axis=0)[:5]
for i in range(5):
    print(f"  {feature_columns[i]:15s}: min={mins[i]:6.3f}, max={maxs[i]:6.3f}")
print()

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 80)
print("PREPROCESSING COMPLETE!")
print("=" * 80)
print()
print("📁 Generated files in:", OUTPUT_DIR.absolute())
print()
print("FOR YOUR TEAM:")
print("  • Random Forest / Decision Trees: Use X_train_unscaled.npy")
print("  • SVM / Logistic Regression: Use X_train_standard.npy")
print("  • Neural Networks: Use X_train_standard.npy or X_train_minmax.npy")
print()
print("LOADING EXAMPLE:")
print("  X_train = np.load('preprocessed_data/X_train_standard.npy')")
print("  y_train = np.load('preprocessed_data/y_train.npy')")
print()
print("=" * 80)

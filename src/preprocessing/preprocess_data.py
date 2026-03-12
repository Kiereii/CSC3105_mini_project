"""
UWB LOS/NLOS Dataset - Data Preprocessing Script
Option B: Core 15 features + CIR samples 730-850
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
import os
import json
import warnings

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = Path("./Dataset/UWB-LOS-NLOS-Data-Set/dataset/Cleaned")
RUN_NAME = os.getenv("RUN_NAME", "split_env_70_15_15_seed42")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
VAL_SIZE = float(os.getenv("VAL_SIZE", "0.15"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.15"))

OUTPUT_DIR = Path("./runs") / RUN_NAME / "preprocessed_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Focused CIR region
CIR_START = 730
CIR_END = 850

print("=" * 80)
print("UWB DATASET PREPROCESSING")
print("=" * 80)
print(
    f"Features: 15 core + {CIR_END - CIR_START} CIR samples = {15 + (CIR_END - CIR_START)} total"
)
print(
    f"Train/Val/Test Split: {int((1 - VAL_SIZE - TEST_SIZE) * 100)}/{int(VAL_SIZE * 100)}/{int(TEST_SIZE * 100)}"
)
print(f"Random Seed: {RANDOM_SEED}")
print("=" * 80)
print()

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("Step 1: Loading dataset...")

csv_files = sorted(DATA_DIR.glob("uwb_cleaned_dataset_part*.csv"))
print(f"Found {len(csv_files)} CSV files")

data_frames = []
for csv_file in csv_files:
    part_df = pd.read_csv(csv_file)
    part_df["__source_file"] = csv_file.name
    data_frames.append(part_df)

df = pd.concat(data_frames, ignore_index=True)
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
X = np.asarray(df[feature_columns].to_numpy(), dtype=np.float64)
y = df["NLOS"].to_numpy(dtype=np.int64)

print(f"✓ Feature matrix shape: {X.shape}")
print(f"✓ Target vector shape: {y.shape}")
print()

# =============================================================================
# STEP 3: TRAIN/VAL/TEST SPLIT (BY ENVIRONMENT)
# =============================================================================
print("Step 3: Splitting data...")

environment_files = sorted(df["__source_file"].unique())
n_env = len(environment_files)

if n_env < 3:
    raise ValueError("Need at least 3 environment files for train/val/test split")

rng = np.random.default_rng(RANDOM_SEED)
shuffled_env_files = environment_files.copy()
rng.shuffle(shuffled_env_files)

n_val_env = max(1, int(round(n_env * VAL_SIZE)))
n_test_env = max(1, int(round(n_env * TEST_SIZE)))

if n_val_env + n_test_env >= n_env:
    n_val_env = 1
    n_test_env = 1

val_envs = list(shuffled_env_files[:n_val_env])
test_envs = list(shuffled_env_files[n_val_env : n_val_env + n_test_env])
train_envs = list(shuffled_env_files[n_val_env + n_test_env :])

if not train_envs:
    raise ValueError("No environments left for training split")

train_mask = df["__source_file"].isin(train_envs)
val_mask = df["__source_file"].isin(val_envs)
test_mask = df["__source_file"].isin(test_envs)

train_idx = df.index[train_mask].to_numpy()
val_idx = df.index[val_mask].to_numpy()
test_idx = df.index[test_mask].to_numpy()

X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Validation set: {X_val.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")
print(f"Training environments: {sorted(train_envs)}")
print(f"Validation environments: {sorted(val_envs)}")
print(f"Test environments: {sorted(test_envs)}")
print()

# Verify balance
train_balance = np.bincount(y_train, minlength=2)
val_balance = np.bincount(y_val, minlength=2)
test_balance = np.bincount(y_test, minlength=2)
print("Class distribution:")
print(
    f"  Training - LOS: {train_balance[0]:,} ({train_balance[0] / len(y_train) * 100:.1f}%), NLOS: {train_balance[1]:,} ({train_balance[1] / len(y_train) * 100:.1f}%)"
)
print(
    f"  Validation - LOS: {val_balance[0]:,} ({val_balance[0] / len(y_val) * 100:.1f}%), NLOS: {val_balance[1]:,} ({val_balance[1] / len(y_val) * 100:.1f}%)"
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
np.save(OUTPUT_DIR / "X_val_unscaled.npy", X_val)
np.save(OUTPUT_DIR / "X_test_unscaled.npy", X_test)
np.save(OUTPUT_DIR / "y_train.npy", y_train)
np.save(OUTPUT_DIR / "y_val.npy", y_val)
np.save(OUTPUT_DIR / "y_test.npy", y_test)
np.save(OUTPUT_DIR / "train_idx.npy", train_idx)
np.save(OUTPUT_DIR / "val_idx.npy", val_idx)
np.save(OUTPUT_DIR / "test_idx.npy", test_idx)

print(f"✓ Saved: X_train_unscaled.npy ({X_train.shape})")
print(f"✓ Saved: X_val_unscaled.npy ({X_val.shape})")
print(f"✓ Saved: X_test_unscaled.npy ({X_test.shape})")
print(f"✓ Saved: y_train.npy ({y_train.shape})")
print(f"✓ Saved: y_val.npy ({y_val.shape})")
print(f"✓ Saved: y_test.npy ({y_test.shape})")
print(f"✓ Saved: train_idx.npy ({train_idx.shape})")
print(f"✓ Saved: val_idx.npy ({val_idx.shape})")
print(f"✓ Saved: test_idx.npy ({test_idx.shape})")
print()

# =============================================================================
# STEP 5: STANDARD SCALER (mean=0, std=1)
# =============================================================================
print("Step 5: Applying StandardScaler...")

scaler_standard = StandardScaler()
X_train_standard = scaler_standard.fit_transform(X_train)
X_val_standard = scaler_standard.transform(X_val)
X_test_standard = scaler_standard.transform(X_test)

# Save scaled data
np.save(OUTPUT_DIR / "X_train_standard.npy", X_train_standard)
np.save(OUTPUT_DIR / "X_val_standard.npy", X_val_standard)
np.save(OUTPUT_DIR / "X_test_standard.npy", X_test_standard)

# Save scaler
with open(OUTPUT_DIR / "scaler_standard.pkl", "wb") as f:
    pickle.dump(scaler_standard, f)

print(f"✓ Saved: X_train_standard.npy")
print(f"✓ Saved: X_val_standard.npy")
print(f"✓ Saved: X_test_standard.npy")
print(f"✓ Saved: scaler_standard.pkl")
print()

# =============================================================================
# STEP 6: SAVE METADATA
# =============================================================================
print("Step 6: Saving metadata...")

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
        f"Training set size: {len(X_train):,} samples ({len(X_train) / len(df) * 100:.1f}%)\n"
    )
    f.write(
        f"Validation set size: {len(X_val):,} samples ({len(X_val) / len(df) * 100:.1f}%)\n"
    )
    f.write(
        f"Test set size: {len(X_test):,} samples ({len(X_test) / len(df) * 100:.1f}%)\n"
    )
    f.write(f"Number of features: {len(feature_columns)}\n")
    f.write(f"  - Core features: {len(core_features)}\n")
    f.write(f"  - CIR features: {len(cir_features)} (samples {CIR_START}-{CIR_END})\n")
    f.write(f"\nTrain environments: {sorted(train_envs)}\n")
    f.write(f"Validation environments: {sorted(val_envs)}\n")
    f.write(f"Test environments: {sorted(test_envs)}\n")
    f.write(f"\nRandom seed: {RANDOM_SEED}\n")
    f.write(f"Run name: {RUN_NAME}\n")
    f.write("Split strategy: Environment-based (group split)\n\n")
    f.write("Generated files:\n")
    f.write("  1. X_train_unscaled.npy / X_val_unscaled.npy / X_test_unscaled.npy\n")
    f.write("  2. X_train_standard.npy / X_val_standard.npy / X_test_standard.npy\n")
    f.write("  3. y_train.npy / y_val.npy / y_test.npy\n")
    f.write("  4. train_idx.npy / val_idx.npy / test_idx.npy\n")
    f.write("  5. scaler_standard.pkl\n")

with open(OUTPUT_DIR / "split_config.json", "w") as f:
    json.dump(
        {
            "run_name": RUN_NAME,
            "random_seed": RANDOM_SEED,
            "val_size": VAL_SIZE,
            "test_size": TEST_SIZE,
            "train_size": 1 - VAL_SIZE - TEST_SIZE,
            "n_total": int(len(df)),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "train_environments": sorted(train_envs),
            "val_environments": sorted(val_envs),
            "test_environments": sorted(test_envs),
        },
        f,
        indent=2,
    )

print(f"✓ Saved: feature_names.txt")
print(f"✓ Saved: preprocessing_info.txt")
print(f"✓ Saved: split_config.json")
print()

# =============================================================================
# STEP 7: VERIFICATION
# =============================================================================
print("Step 7: Verification...")
print()

# Check StandardScaler
print("StandardScaler statistics (first 5 features):")
means = X_train_standard.mean(axis=0)[:5]
stds = X_train_standard.std(axis=0)[:5]
for i in range(5):
    print(f"  {feature_columns[i]:15s}: mean={means[i]:8.3f}, std={stds[i]:6.3f}")
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
print("  • Random Forest / Decision Trees: Use X_train_unscaled.npy")
print("  • SVM / Logistic Regression: Use X_train_standard.npy")
print()
print("LOADING EXAMPLE:")
print("  X_train = np.load('preprocessed_data/X_train_standard.npy')")
print("  y_train = np.load('preprocessed_data/y_train.npy')")
print()
print("=" * 80)

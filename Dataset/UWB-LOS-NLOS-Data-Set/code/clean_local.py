import pandas as pd
import numpy as np
import os

# 1. Setup paths relative to this script
script_dir = os.path.dirname(__file__)
# Logic: Go up from 'code' to Project Root, then into 'dataset/Raw'
raw_dir = os.path.abspath(os.path.join(script_dir, '..', 'dataset', 'Raw'))
cleaned_dir = os.path.abspath(os.path.join(script_dir, '..', 'dataset', 'Cleaned'))

# Ensure the Cleaned directory exists
if not os.path.exists(cleaned_dir):
    os.makedirs(cleaned_dir)

def process_file(input_filename, output_filename):
    # Load from the Raw folder
    raw_path = os.path.join(raw_dir, input_filename)
    df = pd.read_csv(raw_path)
    initial_rows = len(df)
    
    print(f"--- Processing {input_filename} ---")
    print(f"Initial Rows: {initial_rows}")
    
    # --- Step 1: Handle Duplicates ---
    # Removes a row ONLY if EVERY column is identical
    df = df.drop_duplicates()
    dupes_removed = initial_rows - len(df)
    print(f"  - Exact Duplicates Removed: {dupes_removed}")
    
    # --- Step 2: Handle Missing Data (Nulls) ---
    total_nulls_before = df.isnull().sum().sum()
    df = df.dropna()
    print(f"  - Total Null Values Detected: {total_nulls_before}")

    # --- Step 3: Advanced Outlier/Integrity Cleaning ---
    rows_before_outliers = len(df)
    
    # 3a. Ensure Measured Range is possible (must be positive)
    df = df[df['RANGE'] > 0]
    
    # 3b. Filter extreme noise spikes (top 1% of noise)
    if 'STDEV_NOISE' in df.columns:
        threshold = df['STDEV_NOISE'].quantile(0.99)
        df = df[df['STDEV_NOISE'] < threshold]

    outliers_removed = rows_before_outliers - len(df)
    print(f"  - Outliers/Invalid Ranges Removed: {outliers_removed}")
    
    # --- Step 4: Data Transformation ---
    # Ensure Class is integer for ML models
    if 'NLOS' in df.columns:
        df['NLOS'] = df['NLOS'].astype(int)

    # --- Step 5: Feature Extraction (Requirement a.II) ---
    # Calculate Signal-to-Noise Ratio (SNR)
    df['SNR'] = df['FP_AMP1'] / (df['STDEV_NOISE'] + 1e-6)
    # Convert to decibels (dB) for standard signal analysis
    df['SNR_dB'] = 10 * np.log10(df['SNR'].clip(lower=1e-6))
    
    df = df.copy()

    # Save to the Cleaned folder with the new naming convention
    save_path = os.path.join(cleaned_dir, output_filename)
    df.to_csv(save_path, index=False)
    print(f"Final Cleaned Rows: {len(df)}")
    print(f"Total Rows Removed: {initial_rows - len(df)}\n")

# Run for all 7 parts
for i in range(1, 8):
    input_fname = f'uwb_dataset_part{i}.csv'
    output_fname = f'uwb_cleaned_dataset_part{i}.csv'
    
    if os.path.exists(os.path.join(raw_dir, input_fname)):
        process_file(input_fname, output_fname)
    else:
        print(f"Skipping: {input_fname} not found in: {raw_dir}")
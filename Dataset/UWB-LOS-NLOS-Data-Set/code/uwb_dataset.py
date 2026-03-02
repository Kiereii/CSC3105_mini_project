"""
uwb_dataset.py - Shared Data Loader Utility
============================================
Repurposed from the original Klemen Bregar loader (Feb 6, 2017).

This module is the single source of truth for loading UWB data across
all scripts in this project. Both enhanced_visualization.py and the
future train_model.py should import from here instead of duplicating
loading logic.

Usage (in other scripts):
    from uwb_dataset import load_data, CIR_COLUMNS, META_COLUMNS

    df = load_data('Cleaned')   # loads all 7 cleaned CSVs
    df = load_data('Raw')       # loads all 7 raw CSVs
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from numpy import vstack

# ─────────────────────────────────────────────
# CONSTANTS: shared column definitions
# Import these in any script that needs them
# ─────────────────────────────────────────────

# All 1016 CIR tap feature column names
CIR_COLUMNS = [f'CIR{i}' for i in range(1016)]

# Non-CIR metadata/feature columns present in the dataset
META_COLUMNS = ['NLOS', 'RANGE', 'FP_AMP1', 'FP_AMP2', 'STDEV_NOISE', 'CIR_PWR']

# Engineered features added by clean_local.py
ENGINEERED_COLUMNS = ['SNR', 'SNR_dB']

# All non-CIR columns (useful for selecting only metadata quickly)
ALL_META_COLUMNS = META_COLUMNS + ENGINEERED_COLUMNS


# ─────────────────────────────────────────────
# PRIMARY LOADER: used by all other scripts
# ─────────────────────────────────────────────

def load_data(data_type='Cleaned', verbose=True):
    """
    Load all 7 UWB CSV parts into a single combined DataFrame.

    Parameters
    ----------
    data_type : str
        'Cleaned' to load from dataset/Cleaned/
        'Raw'     to load from dataset/Raw/
    verbose : bool
        If True, prints loading progress and summary.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame of all parts with original index reset.

    Raises
    ------
    FileNotFoundError
        If no matching CSV files are found in the expected directory.
    """
    # Resolve path: script is in /code/, data is in /dataset/{Cleaned|Raw}/
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / 'dataset' / data_type

    # Select the correct filename pattern based on data type
    if data_type == 'Cleaned':
        pattern = 'uwb_cleaned_dataset_part*.csv'
    else:
        pattern = 'uwb_dataset_part*.csv'

    csv_paths = sorted(glob.glob(str(data_dir / pattern)))

    if not csv_paths:
        raise FileNotFoundError(
            f"No {data_type} files found at: {data_dir.absolute()}\n"
            f"Expected pattern: {pattern}\n"
            f"Run clean_local.py first if loading Cleaned data."
        )

    if verbose:
        print(f"--- Loading {data_type} Data ---")

    frames = []
    for path in csv_paths:
        part_df = pd.read_csv(path)
        if verbose:
            print(f"  Loaded: {Path(path).name}  ({len(part_df):,} rows)")
        frames.append(part_df)

    df = pd.concat(frames, ignore_index=True)

    if verbose:
        print(f"\nTotal samples loaded : {len(df):,}")
        print(f"Total columns        : {len(df.columns)}")
        los_count  = (df['NLOS'] == 0).sum() if 'NLOS' in df.columns else 'N/A'
        nlos_count = (df['NLOS'] == 1).sum() if 'NLOS' in df.columns else 'N/A'
        print(f"LOS samples          : {los_count:,}")
        print(f"NLOS samples         : {nlos_count:,}")
        print(f"Null values          : {df.isnull().sum().sum()}")
        print("─" * 40)

    return df


def get_cir_columns(df):
    """
    Return only the CIR columns that actually exist in the given DataFrame.
    Useful if CIR trimming has been applied and not all 1016 are present.
    """
    return [col for col in CIR_COLUMNS if col in df.columns]


def get_feature_columns(df, include_engineered=True):
    """
    Return non-CIR feature columns present in the DataFrame.

    Parameters
    ----------
    include_engineered : bool
        If True, includes SNR and SNR_dB columns if present.
    """
    base = [col for col in META_COLUMNS if col in df.columns]
    if include_engineered:
        eng = [col for col in ENGINEERED_COLUMNS if col in df.columns]
        return base + eng
    return base


def split_by_class(df):
    """
    Convenience function: split DataFrame into LOS and NLOS subsets.

    Returns
    -------
    tuple: (los_df, nlos_df)
    """
    if 'NLOS' not in df.columns:
        raise ValueError("DataFrame does not contain 'NLOS' column.")
    return df[df['NLOS'] == 0].copy(), df[df['NLOS'] == 1].copy()


# ─────────────────────────────────────────────
# LEGACY SUPPORT: kept for backwards compatibility
# with the original import_from_files() signature.
# Walks both Raw and Cleaned — use load_data() instead.
# ─────────────────────────────────────────────

def import_from_files():
    """
    LEGACY function from original uwb_dataset.py (Klemen Bregar, 2017).
    Preserved for backwards compatibility only.

    WARNING: This walks ALL subdirectories (Raw AND Cleaned), which will
    double-count data. Use load_data('Raw') or load_data('Cleaned') instead.
    """
    print("[WARNING] import_from_files() is a legacy function.")
    print("          Use load_data('Raw') or load_data('Cleaned') instead.\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    rootdir = os.path.abspath(os.path.join(script_dir, '..', 'dataset'))

    output_arr = None
    for dirpath, dirnames, filenames in os.walk(rootdir):
        for file in filenames:
            if not file.lower().endswith('.csv'):
                continue
            filename = os.path.join(dirpath, file)
            print('Reading:', filename)
            part_df = pd.read_csv(filename, sep=',', header=0)
            input_data = part_df.values
            if output_arr is None:
                output_arr = input_data
            else:
                output_arr = vstack((output_arr, input_data))

    return output_arr if output_arr is not None else []


# ─────────────────────────────────────────────
# STANDALONE: run this file directly to verify
# your dataset loads correctly
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 50)
    print("  UWB Dataset Loader - Verification Run")
    print("=" * 50)

    # Test loading cleaned data
    try:
        df_cleaned = load_data('Cleaned', verbose=True)
        cir_cols = get_cir_columns(df_cleaned)
        feat_cols = get_feature_columns(df_cleaned, include_engineered=True)
        los_df, nlos_df = split_by_class(df_cleaned)

        print(f"\nCIR columns available  : {len(cir_cols)}")
        print(f"Feature columns        : {feat_cols}")
        print(f"LOS subset shape       : {los_df.shape}")
        print(f"NLOS subset shape      : {nlos_df.shape}")
        print("\n[OK] Cleaned data loaded successfully.")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Trying Raw data instead...")

        try:
            df_raw = load_data('Raw', verbose=True)
            print("[OK] Raw data loaded successfully.")
        except FileNotFoundError as e2:
            print(f"[ERROR] {e2}")
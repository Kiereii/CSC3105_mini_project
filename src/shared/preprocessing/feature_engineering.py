"""
Feature Engineering Module for UWB LOS/NLOS Dataset
This module contains functions to extract and engineer features from the raw UWB signals,
particularly focusing on the Channel Impulse Response (CIR) and ratio features to improve
the detection of "clean NLOS" signals.
"""

import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import find_peaks


def compute_rise_time(cir_row):
    """Computes the rise time of the CIR signal (samples from 10% to 90% of peak)."""
    peak_val = np.max(cir_row)
    if peak_val < 1e-10:
        return 0.0
    thresh_10 = 0.1 * peak_val
    thresh_90 = 0.9 * peak_val
    above_10 = np.where(cir_row >= thresh_10)[0]
    above_90 = np.where(cir_row >= thresh_90)[0]
    if len(above_10) == 0 or len(above_90) == 0:
        return 0.0
    return float(above_90[0] - above_10[0])


def count_peaks(cir_row):
    """Counts the number of significant peaks in the CIR signal."""
    peak_val = np.max(cir_row)
    if peak_val < 1e-10:
        return 0.0
    threshold = 0.3 * peak_val
    peaks, _ = find_peaks(cir_row, height=threshold)
    return float(len(peaks))


def engineer_features(X, feature_columns, cir_start=730, cir_end=850):
    """
    Engineers new features from the given feature matrix X.

    Args:
        X: numpy array containing the original feature matrix
        feature_columns: list of strings with the names of the columns in X
        cir_start: starting sample index for the CIR window
        cir_end: ending sample index for the CIR window

    Returns:
        engineered_matrix: numpy array of the new engineered features
        engineered_names: list of the names of the new features
    """

    def _col(name):
        return feature_columns.index(name)

    # Extract raw columns needed for engineering
    fp_amp1 = X[:, _col("FP_AMP1")]
    fp_amp2 = X[:, _col("FP_AMP2")]
    fp_amp3 = X[:, _col("FP_AMP3")]
    stdev_noise = X[:, _col("STDEV_NOISE")]
    cir_pwr = X[:, _col("CIR_PWR")]
    max_noise = X[:, _col("MAX_NOISE")]
    rxpacc = X[:, _col("RXPACC")]
    snr = X[:, _col("SNR")]

    # CIR window
    cir_start_col = _col(f"CIR{cir_start}")
    cir_end_col = _col(f"CIR{cir_end - 1}") + 1
    CIR_matrix = X[:, cir_start_col:cir_end_col]

    engineered_features = []
    engineered_names = []

    # --- Ratio features (target the "clean NLOS" problem) ---

    # 1. FP_AMP_ratio: dominant first path vs second path
    #    LOS has a strong dominant first path; NLOS paths are more spread
    fp_amp_ratio = fp_amp1 / np.maximum(fp_amp2, 1e-10)
    engineered_features.append(fp_amp_ratio)
    engineered_names.append("FP_AMP_ratio")

    # 2. SNR_per_acc: signal quality normalised by preamble accumulation
    snr_per_acc = snr / np.maximum(rxpacc, 1e-10)
    engineered_features.append(snr_per_acc)
    engineered_names.append("SNR_per_acc")

    # 3. signal_to_noise: CIR power over noise standard deviation
    signal_to_noise = cir_pwr / np.maximum(stdev_noise, 1e-10)
    engineered_features.append(signal_to_noise)
    engineered_names.append("signal_to_noise")

    # 4. noise_ratio: impulsiveness of noise (spiky = multipath)
    noise_ratio = max_noise / np.maximum(stdev_noise, 1e-10)
    engineered_features.append(noise_ratio)
    engineered_names.append("noise_ratio")

    # 5. FP_power: total first-path energy
    fp_power = fp_amp1**2 + fp_amp2**2 + fp_amp3**2
    engineered_features.append(fp_power)
    engineered_names.append("FP_power")

    # --- CIR shape features ---

    # 6. CIR_peak: maximum amplitude in the CIR window
    cir_peak = np.max(CIR_matrix, axis=1)
    engineered_features.append(cir_peak)
    engineered_names.append("CIR_peak")

    # 7. CIR_peak_idx: position of peak (shifted = multipath delay)
    cir_peak_idx = np.argmax(CIR_matrix, axis=1).astype(np.float64)
    engineered_features.append(cir_peak_idx)
    engineered_names.append("CIR_peak_idx")

    # 8. CIR_energy: total energy in the CIR window
    cir_energy = np.sum(CIR_matrix**2, axis=1)
    engineered_features.append(cir_energy)
    engineered_names.append("CIR_energy")

    # 9. CIR_kurtosis: peakedness of CIR (sharp=LOS, flat=NLOS)
    cir_kurt = kurtosis(CIR_matrix, axis=1, fisher=True)
    engineered_features.append(cir_kurt)
    engineered_names.append("CIR_kurtosis")

    # 10. CIR_skewness: asymmetry of impulse response
    cir_skew = skew(CIR_matrix, axis=1)
    engineered_features.append(cir_skew)
    engineered_names.append("CIR_skewness")

    # 11. CIR_rise_time: samples from 10% to 90% of peak (slow rise = multipath)
    cir_rise_time = np.array([compute_rise_time(row) for row in CIR_matrix])
    engineered_features.append(cir_rise_time)
    engineered_names.append("CIR_rise_time")

    # 12. CIR_num_peaks: number of significant peaks (more = more multipath = NLOS)
    cir_num_peaks = np.array([count_peaks(row) for row in CIR_matrix])
    engineered_features.append(cir_num_peaks)
    engineered_names.append("CIR_num_peaks")

    # Stack engineered features
    engineered_matrix = np.column_stack(engineered_features)

    return engineered_matrix, engineered_names

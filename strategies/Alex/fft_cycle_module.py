"""
Causal FFT Cycle Detector Module
Standalone functions for detecting dominant market cycles using FFT.
Adapted from Ionut Ciuca's implementation.
"""

import numpy as np
import pandas as pd
from pandas import DataFrame


def _next_power_of_two(n: int) -> int:
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


def _linear_detrend_window(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    L = len(x)
    if L < 3:
        return x - np.nanmean(x)

    j = np.arange(L, dtype=float)
    j_mean = (L - 1) / 2.0
    x_mean = np.nanmean(x)
    jc = j - j_mean
    xc = x - x_mean
    denom = np.sum(jc * jc)
    if denom <= 1e-12:
        return x - x_mean

    b = np.sum(jc * xc) / denom
    a = x_mean - b * j_mean
    return x - (a + b * j)


def _robust_zscore_window(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if np.isfinite(mad) and mad > eps:
        z = (x - med) / (1.4826 * mad + eps)
    else:
        mu = np.nanmean(x)
        sigma = np.nanstd(x, ddof=1)
        if np.isfinite(sigma) and sigma > eps:
            z = (x - mu) / (sigma + eps)
        else:
            z = x - mu

    z = z - np.nanmean(z)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def causal_fft_cycle_period(
    price: pd.Series,
    window_size: int = 100,
    min_period: int = 12,
    max_period: int = 160,
    default_period: int = 80,
    smooth_span: int = 10,
    peak_ratio_min: float = 4.0,
    dominance_min: float = 0.12,
    entropy_max: float = 0.85,
) -> DataFrame:
    """
    Causal FFT cycle detector.
    
    Detects the dominant repeating cycle length in price data using
    rolling-window FFT with quality filters.
    
    Returns DataFrame with columns:
    - cycle_period: smoothed dominant cycle (bars)
    - fft_raw_period: raw detected period
    - fft_peak_ratio: peak vs median power ratio (quality)
    - fft_dominance: peak vs total power ratio (quality)
    - fft_entropy: spectral flatness (quality)
    - fft_valid: True if cycle passes all quality thresholds
    """
    eps = 1e-12
    price = (
        pd.Series(price, copy=True)
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
    )
    index = price.index
    values = price.to_numpy(dtype=float)
    n_rows = len(values)
    L = int(window_size)
    L = max(L, min_period * 2 + 4)

    max_period = int(min(max_period, L // 2))
    min_period = int(max(3, min_period))
    if max_period <= min_period:
        max_period = min_period + 1

    n_fft = _next_power_of_two(L)
    hann = np.hanning(L)
    window_energy = np.mean(hann * hann)
    if not np.isfinite(window_energy) or window_energy <= eps:
        window_energy = 1.0

    freqs = np.fft.rfftfreq(n_fft, d=1.0)
    periods = np.full_like(freqs, np.inf, dtype=float)
    periods[1:] = 1.0 / np.maximum(freqs[1:], eps)
    band_mask = (
        np.isfinite(periods)
        & (periods >= min_period)
        & (periods <= max_period)
    )
    band_idx = np.flatnonzero(band_mask)

    cycle_period = np.full(n_rows, float(default_period), dtype=float)
    raw_period = np.full(n_rows, np.nan, dtype=float)
    peak_ratio = np.full(n_rows, np.nan, dtype=float)
    dominance = np.full(n_rows, np.nan, dtype=float)
    entropy = np.full(n_rows, np.nan, dtype=float)
    valid = np.zeros(n_rows, dtype=bool)

    if len(band_idx) < 3 or n_rows == 0:
        return DataFrame(
            {
                "cycle_period": cycle_period,
                "fft_raw_period": raw_period,
                "fft_peak_ratio": peak_ratio,
                "fft_dominance": dominance,
                "fft_entropy": entropy,
                "fft_valid": valid,
            },
            index=index,
        )

    alpha_period = 2.0 / (float(smooth_span) + 1.0)
    prev_period = float(default_period)

    for i in range(L - 1, n_rows):
        window = values[i - L + 1 : i + 1]
        if (
            len(window) != L
            or not np.all(np.isfinite(window))
            or np.any(window <= 0)
        ):
            cycle_period[i] = prev_period
            continue

        x = np.log(np.clip(window, eps, None))
        x = _linear_detrend_window(x)
        z = _robust_zscore_window(x, eps=eps)
        y = z * hann

        fft_out = np.fft.rfft(y, n=n_fft)
        power = (np.abs(fft_out) ** 2) / (L * window_energy + eps)
        if len(power) > 2:
            if n_fft % 2 == 0:
                power[1:-1] *= 2.0
            else:
                power[1:] *= 2.0
        power = np.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)

        band_power = power[band_idx]
        total_band_power = float(np.sum(band_power))
        if total_band_power <= eps:
            cycle_period[i] = prev_period
            continue

        local_peak_pos = int(np.argmax(band_power))
        k_peak = int(band_idx[local_peak_pos])
        peak_power = float(power[k_peak])
        median_power = float(np.median(band_power))
        pr = peak_power / (median_power + eps)
        dom = peak_power / (total_band_power + eps)
        prob = band_power / (total_band_power + eps)
        ent = -float(np.sum(prob * np.log(prob + eps))) / np.log(len(prob))

        peak_ratio[i] = pr
        dominance[i] = dom
        entropy[i] = ent

        is_valid = (
            pr >= peak_ratio_min
            and dom >= dominance_min
            and ent <= entropy_max
        )
        if is_valid:
            k_hat = float(k_peak)
            if 1 <= k_peak < len(power) - 1:
                left = np.log(power[k_peak - 1] + eps)
                center = np.log(power[k_peak] + eps)
                right = np.log(power[k_peak + 1] + eps)
                denom = left - 2.0 * center + right
                if abs(denom) > eps:
                    delta = 0.5 * (left - right) / denom
                    delta = float(np.clip(delta, -0.5, 0.5))
                    k_hat = float(k_peak) + delta

            detected_period = n_fft / max(k_hat, eps)
            detected_period = float(np.clip(detected_period, min_period, max_period))
            raw_period[i] = detected_period
            valid[i] = True
            smoothed = alpha_period * detected_period + (1.0 - alpha_period) * prev_period
            prev_period = float(np.clip(smoothed, min_period, max_period))
        else:
            prev_period = float(np.clip(prev_period, min_period, max_period))

        cycle_period[i] = prev_period

    cycle_period = np.nan_to_num(
        cycle_period,
        nan=float(default_period),
        posinf=float(default_period),
        neginf=float(default_period),
    )
    return DataFrame(
        {
            "cycle_period": cycle_period,
            "fft_raw_period": raw_period,
            "fft_peak_ratio": peak_ratio,
            "fft_dominance": dominance,
            "fft_entropy": entropy,
            "fft_valid": valid,
        },
        index=index,
    )


def compute_cycle_phase(cycle_period: pd.Series, default_period: float = 80.0) -> pd.Series:
    """
    Compute approximate cycle phase (0.0 to 1.0) where:
    - 0.0 / 1.0 = cycle peak (expected reversal down)
    - 0.5 = cycle trough (expected reversal up)
    
    This uses a simple modulo counter. It's an approximation —
    the true phase requires the FFT complex angle at the dominant frequency.
    """
    phase = pd.Series(np.nan, index=cycle_period.index)
    pos = 0.0
    for i in range(len(cycle_period)):
        cp = cycle_period.iloc[i]
        if np.isfinite(cp) and cp > 0:
            pos = (pos + 1.0) % cp
            phase.iloc[i] = pos / cp
        else:
            pos = (pos + 1.0) % default_period
            phase.iloc[i] = pos / default_period
    return phase

# src/utils/preprocessing.py
import numpy as np
import pandas as pd
import peakutils
from scipy.signal import savgol_filter

from src.utils.spectrum_math import normalize_data


def trim_2theta(df: pd.DataFrame, interval: tuple[float, float]) -> pd.DataFrame:
    """
    Trim spectrum to the given 2θ interval.
    """
    tmin, tmax = interval
    return df.query("@tmin <= two_theta <= @tmax").reset_index(drop=True)


def apply_savgol(intensity: pd.Series | np.ndarray, window_length: int = 17, polyorder: int = 3) -> np.ndarray:
    """
    Savitzky–Golay smoothing for intensity arrays.
    """
    y = np.asarray(intensity, dtype=float)
    if y.size == 0:
        return y
    # window_length must be odd and <= len(y)
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, y.size if y.size % 2 == 1 else y.size - 1)
    if window_length < polyorder + 2:
        return y
    return savgol_filter(y, window_length=window_length, polyorder=polyorder)


def subtract_baseline(
    two_theta: pd.Series | np.ndarray,
    intensity: pd.Series | np.ndarray,
    deg: int = 5,
    max_it: int = 200,
    tol: float = 1e-4,
) -> np.ndarray:
    """
    Baseline subtraction using peakutils.baseline.
    """
    _ = np.asarray(two_theta)  # kept for interface symmetry; not used directly
    y = np.asarray(intensity, dtype=float)
    if y.size == 0:
        return y
    baseline = peakutils.baseline(y, deg=deg, max_it=max_it, tol=tol)
    return y - baseline


def normalize_intensity(intensity: pd.Series | np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Normalize intensities to [0, scale].
    """
    y = normalize_data(np.asarray(intensity, dtype=float))
    return y * scale


# ---- Legacy wrappers kept for backward compatibility ----

def crop_2theta(df: pd.DataFrame, interval_2theta=(20.0, 60.0)) -> pd.DataFrame:
    return trim_2theta(df, interval_2theta)


def apply_savgol_df(
    df: pd.DataFrame,
    window_length: int = 17,
    polyorder: int = 3,
    column: str = "intensity",
) -> pd.DataFrame:
    df = df.copy()
    df[column] = apply_savgol(df[column].values, window_length=window_length, polyorder=polyorder)
    return df


def subtract_baseline_peakutils(
    df: pd.DataFrame,
    deg: int = 5,
    max_it: int = 200,
    tol: float = 1e-4,
    column: str = "intensity",
) -> pd.DataFrame:
    df = df.copy()
    df[column] = subtract_baseline(df["two_theta"].values, df[column].values, deg=deg, max_it=max_it, tol=tol)
    return df


def apply_normalization(
    df: pd.DataFrame,
    column: str = "intensity",
    scale: float = 100.0,
) -> pd.DataFrame:
    df = df.copy()
    df[column] = normalize_intensity(df[column].values, scale=scale)
    return df


def preprocess_xrd(
    df: pd.DataFrame,
    interval_2theta=(20.0, 60.0),
    use_baseline: bool = True,
    use_savgol: bool = True,
    savgol_window: int = 17,
    savgol_polyorder: int = 3,
) -> pd.DataFrame:
    df_proc = crop_2theta(df, interval_2theta)

    if use_baseline:
        df_proc = subtract_baseline_peakutils(df_proc)

    if use_savgol:
        df_proc = apply_savgol_df(
            df_proc, window_length=savgol_window, polyorder=savgol_polyorder
        )

    df_proc = apply_normalization(df_proc)
    return df_proc

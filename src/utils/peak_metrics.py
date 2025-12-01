"""
Peak metrics: area integrals, crystallinity index, and helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_total_area(two_theta: np.ndarray, intensity: np.ndarray) -> float:
    """
    Integral of the whole spectrum using trapezoidal rule.
    """
    if len(two_theta) == 0:
        return 0.0
    return float(np.trapezoid(intensity, two_theta))


def classify_crystalline(fwhm: float, threshold: float = 0.3) -> bool:
    """
    Empirical rule: narrow peaks (FWHM <= threshold) are treated as crystalline.
    """
    return bool(fwhm <= threshold)


def _estimate_peak_area(row: pd.Series) -> float:
    """
    Approximate area for a fitted peak using model-aware formulas.
    """
    amplitude = row.get("amplitude", np.nan)
    model = str(row.get("model", ""))
    height = float(row.get("height", 0) or 0)
    sigma = float(row.get("sigma", 0) or 0)
    gamma = float(row.get("gamma", 0) or 0)
    fwhm = float(row.get("fwhm", 0) or 0)

    if not np.isnan(amplitude):
        return float(abs(amplitude))

    if model == "GaussianModel":
        return abs(height) * abs(sigma) * np.sqrt(2 * np.pi)

    if model == "LorentzianModel":
        width = gamma if gamma else (fwhm / 2 if fwhm else sigma)
        return abs(height) * np.pi * abs(width)

    if model == "VoigtModel":
        width = fwhm if fwhm else (sigma + gamma)
        return abs(height) * abs(width)

    return abs(height) * abs(fwhm)


def compute_ci(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    peak_table: pd.DataFrame,
    fwhm_threshold: float = 0.3,
) -> float:
    """
    Crystallinity Index (CI) = area of crystalline peaks / total area * 100.
    """
    area_total = compute_total_area(two_theta, intensity)
    if area_total <= 0 or peak_table.empty:
        return 0.0

    mask = peak_table["fwhm"] <= fwhm_threshold
    if not mask.any():
        return 0.0

    area_cryst = peak_table.loc[mask].apply(_estimate_peak_area, axis=1).sum()
    ci = (area_cryst / area_total) * 100
    return float(ci)


def select_crystalline_peaks(
    peak_table: pd.DataFrame,
    fwhm_min: float = 0.1,
    fwhm_max: float = 2.0,
    rel_height_min: float = 0.05,
) -> pd.Series:
    """
    Boolean mask for crystalline peaks based on FWHM/height/size sanity checks.
    """
    df = peak_table.copy()
    max_h = df["height"].max() if "height" in df.columns else np.nan

    mask = pd.Series(True, index=df.index)

    if "fwhm" in df.columns:
        mask &= df["fwhm"].between(fwhm_min, fwhm_max)

    if "height" in df.columns and np.isfinite(max_h) and max_h > 0:
        mask &= df["height"] >= rel_height_min * max_h

    if "cryst_size_nm" in df.columns:
        mask &= df["cryst_size_nm"].between(0.5, 200.0)

    return mask


def compute_ci(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    peak_table: pd.DataFrame,
    fwhm_min: float = 0.1,
    fwhm_max: float = 2.0,
    rel_height_min: float = 0.05,
) -> float:
    """
    CI = (area of crystalline peaks / total area) * 100.
    """
    area_total = compute_total_area(two_theta, intensity)
    if area_total <= 0 or peak_table.empty:
        return 0.0

    mask = select_crystalline_peaks(
        peak_table,
        fwhm_min=fwhm_min,
        fwhm_max=fwhm_max,
        rel_height_min=rel_height_min,
    )

    if not mask.any():
        return 0.0

    area_cryst = peak_table.loc[mask].apply(_estimate_peak_area, axis=1).sum()
    ci = (area_cryst / area_total) * 100
    return float(ci)

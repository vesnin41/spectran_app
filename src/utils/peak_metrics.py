"""
Peak metrics: area integrals, crystallinity index, and helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import CONFIG


@dataclass
class FittedPeak:
    """
    Fitted peak with area and phase metadata.
    """

    index: int
    two_theta: float
    fwhm: float
    area: float
    height: float
    is_amorphous: bool = False
    phase_id: str | None = None


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


def fitted_peaks_from_table(peak_table: pd.DataFrame) -> list[FittedPeak]:
    """
    Convert lmfit peak table to structured peaks with area/kind metadata.
    """
    peaks: list[FittedPeak] = []
    if peak_table is None or peak_table.empty:
        return peaks

    for idx, row in peak_table.iterrows():
        area = row["area"] if "area" in row and pd.notna(row["area"]) else _estimate_peak_area(row)
        kind = str(row.get("kind", "")).lower() if "kind" in row else ""
        is_amorph = bool(kind == "amorphous" or row.get("is_amorphous", False))

        peaks.append(
            FittedPeak(
                index=int(idx),
                two_theta=float(row.get("center", np.nan)),
                fwhm=float(row.get("fwhm", np.nan)),
                area=float(area),
                height=float(row.get("height", np.nan)),
                is_amorphous=is_amorph,
                phase_id=row.get("phase_id") if "phase_id" in row else None,
            )
        )
    return peaks


def reclassify_amorphous_by_width(
    peak_table: pd.DataFrame,
    width_factor: float | None = None,
    min_fwhm_amorph: float | None = None,
    min_crystal_size_nm: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Post-fit heuristic: mark very broad peaks as amorphous based on FWHM.

    Returns (updated_df, fwhm_threshold_used).
    """
    df = peak_table.copy()
    width_factor = CONFIG.amorph_width_factor if width_factor is None else width_factor
    min_fwhm_amorph = CONFIG.amorph_fwhm_min if min_fwhm_amorph is None else min_fwhm_amorph
    min_crystal_size_nm = CONFIG.min_crystal_size_nm if min_crystal_size_nm is None else min_crystal_size_nm

    # Ensure flags exist
    if "is_amorphous" not in df.columns:
        df["is_amorphous"] = False
    if "kind" not in df.columns:
        df["kind"] = ""
    if "is_crystalline" not in df.columns:
        df["is_crystalline"] = True
    else:
        df["is_crystalline"] = df["is_crystalline"].fillna(True)

    # Determine baseline FWHM from crystalline-kind peaks
    base = df[df["kind"].astype(str) == "crystalline"]
    base_fwhm = base["fwhm"].replace([np.inf, -np.inf], np.nan)
    med_fwhm = float(base_fwhm.median()) if not base_fwhm.dropna().empty else np.nan
    if not np.isfinite(med_fwhm) or med_fwhm <= 0:
        med_fwhm = float(df["fwhm"].replace([np.inf, -np.inf], np.nan).median())
    if not np.isfinite(med_fwhm) or med_fwhm <= 0:
        med_fwhm = min_fwhm_amorph

    fwhm_threshold = max(min_fwhm_amorph, med_fwhm * width_factor)

    wide_mask = df["fwhm"] >= fwhm_threshold
    df.loc[wide_mask, "is_amorphous"] = True
    df.loc[wide_mask, "is_crystalline"] = False
    df.loc[wide_mask, "kind"] = "amorphous"

    # Update crystalline flag for remaining peaks (drop too small crystallites)
    df.loc[df["is_amorphous"].astype(bool), "is_crystalline"] = False
    size_col = None
    if "crystal_size_nm" in df.columns:
        size_col = "crystal_size_nm"
    elif "cryst_size_nm" in df.columns:
        size_col = "cryst_size_nm"
    if size_col and min_crystal_size_nm is not None:
        df.loc[df[size_col] < min_crystal_size_nm, "is_crystalline"] = False

    return df, fwhm_threshold


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

    # Start mask
    mask = pd.Series(True, index=df.index)

    # Drop amorphous before computing thresholds
    if "is_amorphous" in df.columns:
        mask &= ~df["is_amorphous"].astype(bool)

    # Apply width filter
    if "fwhm" in df.columns:
        mask &= df["fwhm"].between(fwhm_min, fwhm_max)

    # Height relative to max among currently valid peaks
    if "height" in df.columns:
        max_h = df.loc[mask, "height"].max()
        if np.isfinite(max_h) and max_h > 0:
            mask &= df["height"] >= rel_height_min * max_h

    # Sanity check on crystallite size if it doesn't zero everything
    size_col = None
    if "cryst_size_nm" in df.columns:
        size_col = "cryst_size_nm"
    elif "crystal_size_nm" in df.columns:
        size_col = "crystal_size_nm"

    if size_col:
        size_mask = df[size_col].between(0.5, 200.0)
        if size_mask.any():
            mask &= size_mask

    return mask


def compute_ci(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    peak_table: pd.DataFrame,
    fwhm_min: float = 0.1,
    fwhm_max: float = 2.0,
    rel_height_min: float = 0.05,
    fwhm_threshold: float | None = None,
) -> float:
    """
    CI = (area of crystalline peaks / total area) * 100.

    If ``fwhm_threshold`` is provided, it is treated as an upper bound on FWHM
    (kept for backward compatibility) and overrides ``fwhm_min``/``fwhm_max``.
    """
    area_total = compute_total_area(two_theta, intensity)
    if area_total <= 0 or peak_table.empty:
        return 0.0

    if fwhm_threshold is not None:
        fwhm_min, fwhm_max = 0.0, fwhm_threshold

    if "is_crystalline" in peak_table.columns:
        mask = peak_table["is_crystalline"].astype(bool)
    else:
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

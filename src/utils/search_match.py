"""
Search–Match utilities: build reference patterns and compute FoM.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

from src.utils.preprocessing import normalize_intensity, trim_2theta
from src.utils.spectrum_math import normalize_data


def build_ref_pattern_from_cif(
    cif_path: str,
    interval_2theta: tuple[float, float] = (20.0, 60.0),
    phase_id: str | None = None,
) -> pd.DataFrame:
    """
    Generate reference peak list from a CIF file.
    """
    if not os.path.isfile(cif_path):
        raise FileNotFoundError(f"build_ref_pattern_from_cif: file not found: {cif_path}")

    struct = Structure.from_file(cif_path)
    calc = XRDCalculator()
    pattern = calc.get_pattern(struct)

    df = pd.DataFrame({"two_theta": pattern.x, "intensity": pattern.y})
    df = trim_2theta(df, interval_2theta)
    df["intensity"] = normalize_intensity(df["intensity"].values, scale=100.0)

    pid = phase_id or os.path.splitext(os.path.basename(cif_path))[0]
    df["id"] = pid
    df.attrs["phase_id"] = pid
    return df


def _greedy_match_peaks(
    exp_peaks: pd.DataFrame,
    ref_peaks: pd.DataFrame,
    delta_2theta_max: float = 0.2,
) -> pd.DataFrame:
    """
    Greedy one-to-one matching of experimental and reference peaks within tolerance.
    """
    exp = exp_peaks[["center", "height"]].dropna().copy()
    exp["exp_index"] = exp.index
    ref = ref_peaks[["two_theta", "intensity"]].dropna().copy()

    exp["intensity_norm"] = normalize_data(exp["height"].values)
    ref["intensity_norm"] = normalize_data(ref["intensity"].values)

    ref_available = ref.copy()
    matches: list[dict] = []

    # Process experimental peaks from most intense to least
    exp_sorted = exp.sort_values("intensity_norm", ascending=False)
    for _, row in exp_sorted.iterrows():
        x0 = float(row["center"])
        mask = np.abs(ref_available["two_theta"] - x0) <= delta_2theta_max
        candidates = ref_available.loc[mask]
        if candidates.empty:
            continue

        dtheta = np.abs(candidates["two_theta"] - x0)
        idx = dtheta.idxmin()
        ref_row = candidates.loc[idx]

        matches.append(
            dict(
                exp_index=int(row["exp_index"]),
                center_exp=x0,
                height_exp=float(row["height"]),
                intensity_exp_norm=float(row["intensity_norm"]),
                center_ref=float(ref_row["two_theta"]),
                intensity_ref=float(ref_row["intensity"]),
                intensity_ref_norm=float(ref_row["intensity_norm"]),
                dtheta=float(abs(ref_row["two_theta"] - x0)),
            )
        )
        ref_available = ref_available.drop(index=idx)

    return pd.DataFrame(matches)


def compute_fom(
    exp_peaks: pd.DataFrame,
    ref_peaks: pd.DataFrame,
    delta_2theta_max: float = 0.2,
) -> float:
    """
    Compute Search–Match FoM based on Δ2θ, ΔI, and intensity distributions.
    Lower values correspond to better matches.
    """
    if exp_peaks.empty or ref_peaks.empty:
        return float("inf")

    matches = _greedy_match_peaks(exp_peaks, ref_peaks, delta_2theta_max=delta_2theta_max)
    if matches.empty:
        return float("inf")

    theta_term = matches["dtheta"].mean() / delta_2theta_max
    theta_term = float(np.clip(theta_term, 0.0, 1.0))

    intensity_term = np.abs(matches["intensity_exp_norm"] - matches["intensity_ref_norm"]).mean()
    intensity_term = float(np.clip(intensity_term, 0.0, 1.0))

    exp_sorted = np.sort(matches["intensity_exp_norm"].values)[::-1]
    ref_sorted = np.sort(matches["intensity_ref_norm"].values)[::-1]
    distrib_len = min(len(exp_sorted), len(ref_sorted))
    distribution_term = (
        np.abs(exp_sorted[:distrib_len] - ref_sorted[:distrib_len]).mean() if distrib_len else 1.0
    )
    distribution_term = float(np.clip(distribution_term, 0.0, 1.0))

    fom = (theta_term + intensity_term + distribution_term) / 3.0
    return float(fom)


def search_match_all_phases(
    exp_peaks: pd.DataFrame,
    ref_db: Iterable[pd.DataFrame],
    delta_2theta_max: float = 0.2,
) -> pd.DataFrame:
    """
    Evaluate FoM for all reference phases.
    """
    rows = []
    for idx, ref_df in enumerate(ref_db):
        phase_id = ref_df.attrs.get("phase_id") or ref_df.get("id")
        if isinstance(phase_id, pd.Series):
            phase_id = phase_id.iloc[0]
        if phase_id is None:
            phase_id = f"phase_{idx}"

        matches = _greedy_match_peaks(exp_peaks, ref_df, delta_2theta_max=delta_2theta_max)
        fom = compute_fom(exp_peaks, ref_df, delta_2theta_max=delta_2theta_max)
        rows.append(
            dict(
                phase_id=phase_id,
                FoM=fom,
                n_matched_peaks=len(matches),
                comment="" if len(matches) else "no matched peaks",
            )
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("FoM", ascending=True).reset_index(drop=True)
    return result


def annotate_peaks_with_phases(
    exp_peaks: pd.DataFrame,
    ref_db: Iterable[pd.DataFrame],
    delta_2theta_max: float = 0.2,
) -> pd.DataFrame:
    """
    Annotate each experimental peak with the best matching phase by minimal dtheta/intensity diff.

    Returns DataFrame with columns:
        exp_index, center_exp, phase_id, dtheta, intensity_diff
    """
    rows: list[dict] = []

    for idx_phase, ref_df in enumerate(ref_db):
        phase_id = ref_df.attrs.get("phase_id") or ref_df.get("id")
        if isinstance(phase_id, pd.Series):
            phase_id = phase_id.iloc[0]
        if phase_id is None:
            phase_id = f"phase_{idx_phase}"

        matches = _greedy_match_peaks(exp_peaks, ref_df, delta_2theta_max=delta_2theta_max)
        if matches.empty:
            continue

        for _, mrow in matches.iterrows():
            rows.append(
                dict(
                    exp_index=int(mrow["exp_index"]) if "exp_index" in mrow else None,
                    center_exp=mrow["center_exp"],
                    phase_id=phase_id,
                    dtheta=mrow["dtheta"],
                    intensity_diff=float(
                        abs(mrow["intensity_exp_norm"] - mrow["intensity_ref_norm"])
                    ),
                )
            )

    if not rows:
        return pd.DataFrame()

    all_matches = pd.DataFrame(rows)
    all_matches = all_matches.sort_values(
        by=["exp_index", "dtheta", "intensity_diff"], ascending=[True, True, True]
    )
    best = all_matches.groupby("exp_index", as_index=False).first()
    return best


def annotate_peaks_with_best_phase(
    exp_peaks: pd.DataFrame,
    ref_db: Iterable[pd.DataFrame],
    delta_2theta_max: float = 0.2,
) -> pd.DataFrame:
    """
    Return exp_peaks copy with phase_id/dtheta_match/intensity_diff columns.
    """
    matches = annotate_peaks_with_phases(exp_peaks, ref_db, delta_2theta_max=delta_2theta_max)
    exp_peaks = exp_peaks.copy()
    exp_peaks["phase_id"] = None
    exp_peaks["dtheta_match"] = np.nan
    exp_peaks["intensity_diff"] = np.nan

    for _, row in matches.iterrows():
        idx = row["exp_index"]
        if idx in exp_peaks.index:
            exp_peaks.at[idx, "phase_id"] = row["phase_id"]
            exp_peaks.at[idx, "dtheta_match"] = row["dtheta"]
            exp_peaks.at[idx, "intensity_diff"] = row["intensity_diff"]
    return exp_peaks

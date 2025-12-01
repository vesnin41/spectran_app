import os
from typing import Optional, Sequence

import pandas as pd
import pymatgen as mp
from pymatgen.analysis.diffraction.xrd import XRDCalculator

from src.utils.preprocessing import apply_savgol, normalize_intensity, trim_2theta


def spectrum_from_csv(
    file_path: str,
    sep: str = ",",
    has_header: bool = False,
    theta_col: int | str = 0,
    int_col: int | str = 1,
    interval_2theta: Optional[tuple[float, float]] = None,
    filter: bool = False,
    window_length: int = 17,
    polyorder: int = 3,
    normalize: bool = False,
    normalize_scale: float = 1.0,
) -> pd.DataFrame:
    """
    Load raw spectrum from CSV into standardized DataFrame.

    Default CSV format (no header):
        col0: two_theta (degrees)
        col1: intensity (arb. units)

    Optional args allow handling other delimiters, headers, column order,
    light smoothing, normalization, and cropping by 2θ.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"spectrum_from_csv: file not found: {file_path}")

    header = 0 if has_header else None
    df = pd.read_csv(file_path, sep=sep, header=header)

    # Select and rename columns
    try:
        df = df[[theta_col, int_col]].copy()
    except KeyError as exc:
        raise KeyError(
            f"spectrum_from_csv: cannot find columns '{theta_col}'/'{int_col}' in {file_path}"
        ) from exc

    df.columns = ["two_theta", "intensity"]

    # Optional quick preprocessing (kept lightweight)
    if filter:
        df["intensity"] = apply_savgol(
            df["intensity"].values, window_length=window_length, polyorder=polyorder
        )

    if normalize:
        df["intensity"] = normalize_intensity(df["intensity"].values, scale=normalize_scale)

    if interval_2theta is not None:
        df = trim_2theta(df, interval_2theta)

    return df


def phase_from_cif(cif_file_path: str) -> pd.DataFrame:
    """
    Load reference pattern from CIF using pymatgen.

    Returns raw (two_theta, intensity) without cropping or normalization.
    """
    if not os.path.isfile(cif_file_path):
        raise FileNotFoundError(f"phase_from_cif: file not found: {cif_file_path}")

    cif_struct = mp.Structure.from_file(cif_file_path)
    calc = XRDCalculator()
    pattern = calc.get_pattern(cif_struct)

    df = pd.DataFrame({"two_theta": pattern.x, "intensity": pattern.y})
    return df


def phase_from_csv(
    csv_file_path: str,
    sep: str = ",",
    usecols: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """
    Load phase pattern from CSV containing (h, k, l, d_hkl, intensity).

    Returns raw data; does not compute 2theta or normalize—these are preprocessing steps.
    """
    if not os.path.isfile(csv_file_path):
        raise FileNotFoundError(f"phase_from_csv: file not found: {csv_file_path}")

    df = pd.read_csv(csv_file_path, sep=sep, usecols=usecols)
    df.columns = ["h", "k", "l", "d_hkl", "intensity"]
    return df

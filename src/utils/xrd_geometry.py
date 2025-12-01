"""
Geometry helpers for XRD: Bragg's law and Scherrer equation.
"""

from __future__ import annotations

import numpy as np


def get_d_hkl(center_deg: float, wavelength_ang: float = 1.54060) -> float:
    """
    Compute interplanar spacing d_hkl (Å) from peak position (2θ, degrees).
    """
    theta = np.deg2rad(center_deg / 2.0)
    if theta == 0:
        return np.nan
    return float(wavelength_ang / (2 * np.sin(theta)))


def get_crystallite_size_scherrer(
    fwhm_deg: float,
    center_deg: float,
    wavelength_nm: float = 0.154060,
    k: float = 0.9,
) -> float:
    """
    Crystallite size (nm) via the Scherrer equation.
    """
    theta = np.deg2rad(center_deg / 2.0)
    beta = np.deg2rad(fwhm_deg)
    if beta == 0:
        return np.nan
    return float(k * wavelength_nm / (beta * np.cos(theta)))


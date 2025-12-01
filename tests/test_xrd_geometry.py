import numpy as np

from src.utils.spectrum_math import bragg_2theta_from_d
from src.utils.xrd_geometry import get_crystallite_size_scherrer, get_d_hkl


def test_get_d_hkl_roundtrip():
    center_deg = 30.0
    d = get_d_hkl(center_deg)
    two_theta_back = bragg_2theta_from_d(d)
    assert np.isclose(two_theta_back, center_deg, atol=1e-6)


def test_crystallite_size_positive_and_nan():
    size = get_crystallite_size_scherrer(fwhm_deg=0.2, center_deg=30.0)
    assert size > 0

    size_nan = get_crystallite_size_scherrer(fwhm_deg=0.0, center_deg=30.0)
    assert np.isnan(size_nan)

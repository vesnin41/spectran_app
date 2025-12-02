import numpy as np
import pandas as pd

from src.utils.peak_metrics import classify_crystalline, compute_ci, compute_total_area
from src.utils.peak_metrics import FittedPeak, fitted_peaks_from_table, select_crystalline_peaks


def test_compute_total_area_constant():
    two_theta = np.linspace(0, 10, 6)
    intensity = np.full_like(two_theta, 2.0)
    area = compute_total_area(two_theta, intensity)
    expected = 2.0 * (two_theta[-1] - two_theta[0])
    assert np.isclose(area, expected)


def test_classify_crystalline_threshold():
    assert classify_crystalline(0.2, threshold=0.3) is True
    assert classify_crystalline(0.5, threshold=0.3) is False


def test_compute_ci_reacts_to_threshold():
    two_theta = np.linspace(20, 40, 200)
    intensity = np.ones_like(two_theta)

    peak_table = pd.DataFrame(
        [
            {"model": "GaussianModel", "amplitude": 10.0, "height": 10.0, "sigma": 0.1, "gamma": 0.0, "fwhm": 0.15},
            {"model": "GaussianModel", "amplitude": 5.0, "height": 5.0, "sigma": 0.5, "gamma": 0.0, "fwhm": 0.6},
        ]
    )

    ci_loose = compute_ci(two_theta, intensity, peak_table, fwhm_max=0.7)
    ci_strict = compute_ci(two_theta, intensity, peak_table, fwhm_max=0.2)

    assert 0 < ci_loose <= 100
    assert 0 <= ci_strict < ci_loose


def test_compute_ci_threshold_alias():
    two_theta = np.linspace(0, 1, 50)
    intensity = np.ones_like(two_theta)

    peak_table = pd.DataFrame(
        [
            {"model": "GaussianModel", "amplitude": 4.0, "height": 4.0, "sigma": 0.05, "gamma": 0.0, "fwhm": 0.12},
            {"model": "GaussianModel", "amplitude": 1.0, "height": 1.0, "sigma": 0.2, "gamma": 0.0, "fwhm": 0.25},
            {"model": "GaussianModel", "amplitude": 0.5, "height": 0.5, "sigma": 0.4, "gamma": 0.0, "fwhm": 0.6},
        ]
    )

    ci_threshold = compute_ci(two_theta, intensity, peak_table, fwhm_threshold=0.3)
    ci_range = compute_ci(two_theta, intensity, peak_table, fwhm_min=0.0, fwhm_max=0.3)

    assert ci_threshold > 0
    assert np.isclose(ci_threshold, ci_range)


def test_fitted_peaks_from_table_marks_amorphous():
    peak_table = pd.DataFrame(
        [
            {"center": 10.0, "fwhm": 0.2, "height": 5.0, "area": 1.0, "kind": "crystalline"},
            {"center": 20.0, "fwhm": 0.5, "height": 2.0, "area": 2.0, "kind": "amorphous"},
        ]
    )
    peaks = fitted_peaks_from_table(peak_table)
    assert len(peaks) == 2
    assert peaks[0].is_amorphous is False
    assert peaks[1].is_amorphous is True


def test_select_crystalline_respects_is_amorphous_and_size_guard():
    df = pd.DataFrame(
        [
            {"height": 10.0, "fwhm": 0.5, "cryst_size_nm": 0.1, "is_amorphous": False},
            {"height": 8.0, "fwhm": 0.6, "cryst_size_nm": 0.2, "is_amorphous": True},
            {"height": 9.0, "fwhm": 0.7, "cryst_size_nm": 0.3, "is_amorphous": False},
        ]
    )
    # Size filter would zero all; guard should keep fwhm/height-based mask
    mask = select_crystalline_peaks(df, fwhm_min=0.1, fwhm_max=1.0, rel_height_min=0.5)
    assert mask.any()
    assert bool(mask.iloc[1]) is False  # amorphous excluded


def test_select_crystalline_filters_unrealistic_large_sizes():
    df = pd.DataFrame(
        [
            {"height": 10.0, "fwhm": 0.5, "crystal_size_nm": 50.0, "is_amorphous": False},
            {"height": 9.0, "fwhm": 0.5, "crystal_size_nm": 5000.0, "is_amorphous": False},
        ]
    )
    mask = select_crystalline_peaks(df, fwhm_min=0.1, fwhm_max=1.0, rel_height_min=0.1)
    assert bool(mask.iloc[0]) is True
    assert bool(mask.iloc[1]) is False

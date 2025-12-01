import numpy as np
import pandas as pd

from src.utils.peak_metrics import classify_crystalline, compute_ci, compute_total_area


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

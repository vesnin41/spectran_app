import numpy as np
import pandas as pd

from src.utils.search_match import _greedy_match_peaks, compute_fom


def _make_peaks(centers, intensities):
    return pd.DataFrame({"center": centers, "height": intensities})


def _make_ref(two_theta, intensities):
    return pd.DataFrame({"two_theta": two_theta, "intensity": intensities})


def test_greedy_match_peaks_counts():
    exp = _make_peaks(np.array([30.0, 32.0, 34.0]), np.array([10.0, 8.0, 6.0]))
    ref = _make_ref(np.array([30.05, 31.95, 40.0]), np.array([9.0, 7.0, 1.0]))

    matches = _greedy_match_peaks(exp, ref, delta_2theta_max=0.2)

    assert len(matches) == 2 or len(matches) == 3

    far_ref = _make_ref(np.array([50.0, 52.0]), np.array([5.0, 4.0]))
    matches_far = _greedy_match_peaks(exp, far_ref, delta_2theta_max=0.1)
    assert len(matches_far) == 0


def test_compute_fom_ordering():
    exp = _make_peaks(np.array([30.0, 32.0, 34.0]), np.array([10.0, 8.0, 6.0]))
    ref_good = _make_ref(np.array([30.05, 31.95, 34.1]), np.array([9.0, 7.0, 6.0]))
    ref_bad = _make_ref(np.array([40.0, 42.0, 44.0]), np.array([1.0, 1.0, 1.0]))

    fom_good = compute_fom(exp, ref_good, delta_2theta_max=0.2)
    fom_bad = compute_fom(exp, ref_bad, delta_2theta_max=0.2)

    assert fom_good >= 0
    assert fom_bad >= 0
    assert fom_good < fom_bad

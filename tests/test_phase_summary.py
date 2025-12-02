import numpy as np

from src.analysis.phase_summary import aggregate_phase_results, compute_area_fractions, PhaseResult
from src.utils.peak_metrics import FittedPeak


def test_phase_aggregation_and_fractions():
    peaks = [
        FittedPeak(index=0, two_theta=10.0, fwhm=0.2, area=5.0, height=5.0, phase_id="A"),
        FittedPeak(index=1, two_theta=12.0, fwhm=0.25, area=3.0, height=3.0, phase_id="A"),
        FittedPeak(index=2, two_theta=20.0, fwhm=0.3, area=4.0, height=4.0, phase_id="B"),
        FittedPeak(index=3, two_theta=30.0, fwhm=0.6, area=2.0, height=2.0, is_amorphous=True, phase_id="amorphous"),
    ]

    phase_results = aggregate_phase_results(peaks)
    phases = {pr.phase_id: pr for pr in phase_results}

    assert "A" in phases and "B" in phases and "amorphous" in phases
    assert np.isclose(phases["A"].area_total, 8.0)
    assert phases["A"].n_peaks == 2

    fractions = compute_area_fractions(peaks)
    assert np.isclose(fractions["area_total"], 14.0)
    assert np.isclose(fractions["area_cryst"], 12.0)
    assert np.isclose(fractions["area_amorph"], 2.0)
    assert np.isclose(fractions["x_cryst"] + fractions["x_amorph"], 1.0)


def test_phase_result_mean_crystallite_size_increases_with_narrower_peaks():
    # Narrower peaks -> larger size
    pk_wide = FittedPeak(index=0, two_theta=30.0, fwhm=0.4, area=1.0, height=2.0, phase_id="A")
    pk_narrow = FittedPeak(index=1, two_theta=30.0, fwhm=0.2, area=1.0, height=2.0, phase_id="A")
    pr = PhaseResult(phase_id="A", peaks=[pk_wide, pk_narrow])

    size = pr.mean_crystallite_size()
    assert size is not None
    pr_wider = PhaseResult(phase_id="A", peaks=[pk_wide])
    pr_narrower = PhaseResult(phase_id="A", peaks=[pk_narrow])
    assert pr_narrower.mean_crystallite_size() > pr_wider.mean_crystallite_size()

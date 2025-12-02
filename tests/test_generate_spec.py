import numpy as np

from src.fitting.generate_spec import Peak, build_model_from_peaks, detect_peaks


def test_detect_peaks_returns_positions_and_width():
    x = np.linspace(10, 20, 500)
    y = (
        np.exp(-0.5 * ((x - 12.0) / 0.1) ** 2)
        + 1.5 * np.exp(-0.5 * ((x - 17.0) / 0.15) ** 2)
    )

    peaks = detect_peaks(x, y, width_range=(3, 80), rel_height_min=0.1, prominence_rel=0.05)
    assert len(peaks) == 2

    centers = sorted([p.two_theta for p in peaks])
    assert np.isclose(centers[0], 12.0, atol=0.1)
    assert np.isclose(centers[1], 17.0, atol=0.1)
    assert all(p.fwhm_est > 0 for p in peaks)


def test_build_model_from_peaks_counts_extra_and_amorph():
    peaks = [
        Peak(index=0, two_theta=10.0, height=5.0, fwhm_est=0.2),
        Peak(index=1, two_theta=20.0, height=2.5, fwhm_est=0.25),
    ]
    models = build_model_from_peaks(
        peaks,
        model_type="GaussianModel",
        two_theta=np.linspace(5, 25, 200),
        intensity=np.ones(200),
        extra_components=3,
        amorph_components=1,
    )

    assert len(models) == 5  # 2 peaks + 3 extra
    amorph_sigma = [m["params"]["sigma"] for m in models if m.get("param_hints")]
    narrow_sigma = [m["params"]["sigma"] for m in models if not m.get("param_hints")]
    assert amorph_sigma and max(amorph_sigma) > max(narrow_sigma)

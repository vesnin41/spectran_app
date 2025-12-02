import pandas as pd

from src.utils.peak_metrics import reclassify_amorphous_by_width


def test_reclassify_amorphous_by_width_marks_wide_peaks():
    df = pd.DataFrame(
        [
            {"fwhm": 0.5, "kind": "crystalline", "is_amorphous": False, "crystal_size_nm": 10.0},
            {"fwhm": 0.6, "kind": "crystalline", "is_amorphous": False, "crystal_size_nm": 12.0},
            {"fwhm": 3.0, "kind": "crystalline", "is_amorphous": False, "crystal_size_nm": 1.0},
        ]
    )

    updated, fwhm_thr = reclassify_amorphous_by_width(df, width_factor=2.0, min_fwhm_amorph=1.0)
    assert fwhm_thr >= 1.0
    assert bool(updated.loc[2, "is_amorphous"]) is True
    assert bool(updated.loc[2, "is_crystalline"]) is False
    assert bool(updated.loc[0, "is_crystalline"]) is True
    assert bool(updated.loc[1, "is_crystalline"]) is True

import numpy as np
import pandas as pd

from src.fitting import generate_spec
from src.utils.peak_profiles import gaussian


def make_synthetic_df():
    x = np.linspace(20, 60, 2000)
    y = gaussian(x, amplitude=100, center=30, sigma=0.1) + gaussian(
        x, amplitude=80, center=45, sigma=0.15
    )
    y += np.random.default_rng(0).normal(0, 0.5, size=x.size)
    return pd.DataFrame({"two_theta": x, "intensity": y})


def test_default_spec_shapes():
    df = make_synthetic_df()
    models_arr = [{"type": "GaussianModel"} for _ in range(5)]

    spec = generate_spec.default_spec(df, models_arr)

    assert np.array_equal(spec["x"], df["two_theta"].values)
    assert np.array_equal(spec["y"], df["intensity"].values)
    assert len(spec["model"]) == 5


def test_update_spec_from_peaks_fills_params():
    df = make_synthetic_df()
    models_arr = [{"type": "GaussianModel"} for _ in range(5)]
    spec = generate_spec.default_spec(df, models_arr)

    spec, peaks = generate_spec.update_spec_from_peaks(
        spec,
        model_indicies=list(range(5)),
        peak_widths=(3, 40),
        method="find_peaks",
    )

    assert len(peaks) > 0
    for idx, model in enumerate(spec["model"]):
        params = model.get("params", {})
        if idx < len(peaks):
            assert "center" in params and "sigma" in params and "height" in params
        else:
            assert params == {}

import numpy as np
import pandas as pd

from src.fitting import fit_model, generate_spec
from src.utils.peak_profiles import gaussian


def make_synthetic_df():
    x = np.linspace(20, 60, 2000)
    y = gaussian(x, amplitude=100, center=30.0, sigma=0.1) + gaussian(
        x, amplitude=70, center=45.0, sigma=0.15
    )
    y += np.random.default_rng(1).normal(0, 0.5, size=x.size)
    return pd.DataFrame({"two_theta": x, "intensity": y}), [30.0, 45.0]


def test_model_fit_and_best_values():
    df, centers_true = make_synthetic_df()
    models_arr = [{"type": "GaussianModel"} for _ in range(4)]
    spec = generate_spec.default_spec(df, models_arr)
    spec, peaks = generate_spec.update_spec_from_peaks(
        spec,
        model_indicies=list(range(4)),
        peak_widths=(3, 40),
        method="find_peaks",
    )
    assert len(peaks) > 0

    model = fit_model.Model(spec)
    output = model.fit()
    assert output is not None

    peak_table = model.print_best_values()
    assert not peak_table.empty

    centers_est = peak_table["center"].dropna().values
    assert any(np.isclose(c, centers_true[0], atol=0.2) for c in centers_est)
    assert any(np.isclose(c, centers_true[1], atol=0.2) for c in centers_est)

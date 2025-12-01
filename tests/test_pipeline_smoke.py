import numpy as np
import pandas as pd

from src.fitting import fit_model, generate_spec
from src.reading.data_from import spectrum_from_csv
from src.utils.peak_metrics import compute_ci
from src.utils.preprocessing import (
    apply_savgol,
    normalize_intensity,
    subtract_baseline,
    trim_2theta,
)


def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma**2))


def test_pipeline_smoke(tmp_path):
    rng = np.random.default_rng(42)
    x = np.linspace(20, 60, 1000)
    y = gaussian(x, 100, 30, 0.1) + gaussian(x, 80, 45, 0.15)
    y += rng.normal(0, 0.5, size=x.size)

    path = tmp_path / "synthetic.csv"
    pd.DataFrame({"two_theta": x, "intensity": y}).to_csv(path, header=False, index=False)

    df_raw = spectrum_from_csv(str(path))
    df = trim_2theta(df_raw, interval=(20, 60))
    df["intensity"] = apply_savgol(df["intensity"], window_length=31, polyorder=3)
    df["intensity"] = subtract_baseline(df["two_theta"], df["intensity"])
    df["intensity"] = normalize_intensity(df["intensity"]) * 100

    models_arr = [{"type": "GaussianModel"} for _ in range(10)]
    spec = generate_spec.default_spec(df, models_arr)
    spec, peaks_found = generate_spec.update_spec_from_peaks(
        spec,
        list(range(10)),
        peak_widths=(3, 40),
        method="find_peaks",
    )

    model = fit_model.Model(spec)
    output = model.fit()
    peak_table = model.print_best_values()
    ci = compute_ci(df["two_theta"].values, df["intensity"].values, peak_table)

    assert output is not None
    assert not peak_table.empty
    assert len(peaks_found) > 0
    assert 0 <= ci <= 100

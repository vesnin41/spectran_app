import numpy as np
import pandas as pd

from src.utils.preprocessing import (
    apply_savgol,
    normalize_intensity,
    subtract_baseline,
    trim_2theta,
)


def test_trim_2theta():
    df = pd.DataFrame({"two_theta": np.linspace(10, 70, 13), "intensity": np.arange(13)})
    trimmed = trim_2theta(df, interval=(20, 60))
    assert trimmed["two_theta"].between(20, 60).all()
    assert len(trimmed) < len(df)


def test_apply_savgol_reduces_noise():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 2 * np.pi, 200)
    clean = np.sin(x)
    noise = rng.normal(0, 0.2, size=clean.size)
    noisy = clean + noise

    filtered = apply_savgol(noisy, window_length=31, polyorder=3)

    assert len(filtered) == len(noisy)
    assert np.var(filtered - clean) < np.var(noisy - clean)


def test_subtract_baseline_on_linear_trend():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 100)
    trend = 0.5 * x + 5
    noise = rng.normal(0, 0.1, size=x.size)
    signal = trend + noise

    corrected = subtract_baseline(x, signal, deg=1)

    assert abs(np.mean(corrected)) < abs(np.mean(signal))


def test_normalize_intensity():
    arr = np.array([1, 2, 3], dtype=float)
    norm = normalize_intensity(arr, scale=2.0)
    expected = np.array([0.0, 1.0, 2.0])
    assert np.allclose(norm, expected)

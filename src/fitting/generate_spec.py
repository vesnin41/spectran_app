# src/fitting/generate_spec.py

import numpy as np
from scipy import signal
from typing import Tuple


def _detect_peaks_find_peaks(
    x: np.ndarray,
    y: np.ndarray,
    width_range: Tuple[int, int] | None = None,
    rel_height_min: float = 0.05,
    prominence: float | None = None,
    prominence_rel: float | None = None,
    distance: int | None = None,
) -> np.ndarray:
    """
    Robust peak detection for XRD using scipy.signal.find_peaks.

    Parameters
    ----------
    x, y : array-like
        two_theta and intensity arrays.
    width_range : (min_width, max_width) in points.
    rel_height_min : float
        Minimum relative height (fraction of y.max()).
    prominence : float | None
        Minimum prominence for peaks.
    prominence_rel : float | None
        Relative prominence (fraction of y.max()); used if prominence is None.
    distance : int | None
        Minimum distance between peaks (in points).
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return np.array([], dtype=int)

    y_max = float(np.max(y))
    if y_max <= 0:
        return np.array([], dtype=int)

    height = rel_height_min * y_max if rel_height_min is not None else None
    prom = prominence if prominence is not None else (
        prominence_rel * y_max if prominence_rel is not None else None
    )

    width = None
    if width_range is not None:
        wmin, wmax = width_range
        width = (max(1, int(wmin)), max(1, int(wmax)))

    peaks, _ = signal.find_peaks(
        y,
        height=height,
        prominence=prom,
        width=width,
        distance=distance,
    )

    return peaks.astype(int)


def default_spec(df, models_arr):
    """
    Create a specification dictionary ('spec') from DataFrame.

    Expected columns:
        df['two_theta'] : 2θ values (degrees)
        df['intensity'] : intensities

    Returns
    -------
    spec : dict with keys:
        - 'x': np.ndarray
        - 'y': np.ndarray
        - 'model': list of model definitions
    """
    return {
        "x": df["two_theta"].values,
        "y": df["intensity"].values,
        "model": models_arr,
    }


def update_spec_from_peaks(
    spec,
    model_indicies,
    peak_widths=(7, 80),
    method: str = "find_peaks",
    rel_height_min: float = 0.05,
    prominence: float | None = None,
    prominence_rel: float | None = 0.02,
    distance: int | None = None,
):
    """
    Find peaks and initialize peak parameters for lmfit models.
    """
    x = np.asarray(spec["x"], dtype=float)
    y = np.asarray(spec["y"], dtype=float)

    if method == "find_peaks":
        if isinstance(peak_widths, tuple) and len(peak_widths) == 2:
            width_range = (int(peak_widths[0]), int(peak_widths[1]))
        else:
            width_range = None

        dist_pts = distance
        if dist_pts is None and width_range is not None:
            dist_pts = max(1, width_range[0])

        peaks = _detect_peaks_find_peaks(
            x,
            y,
            width_range=width_range,
            rel_height_min=rel_height_min,
            prominence=prominence,
            prominence_rel=prominence_rel,
            distance=dist_pts,
        )

    elif method == "cwt":
        if isinstance(peak_widths, np.ndarray):
            widths = peak_widths
        elif isinstance(peak_widths, tuple) and len(peak_widths) == 2:
            widths = np.arange(peak_widths[0], peak_widths[1])
        else:
            widths = np.asarray(peak_widths)

        peaks = signal.find_peaks_cwt(y, widths=widths)
        peaks = np.asarray(peaks, dtype=int)

        if y.size:
            y_max = float(np.max(y))
            thr = rel_height_min * y_max
            peaks = peaks[y[peaks] >= thr]
    else:
        raise ValueError(f"Unknown peak detection method: {method!r}")

    if peaks.size == 0:
        print("No peaks found.")
        return spec, peaks

    # Sort peaks by intensity (descending)
    order = np.argsort(y[peaks])[::-1]
    peak_indices = peaks[order][: len(model_indicies)]

    # Estimate sigma based on median width
    if len(x) > 1:
        dx = (x.max() - x.min()) / (len(x) - 1)
    else:
        dx = 1.0
    if isinstance(peak_widths, tuple) and len(peak_widths) == 2:
        min_width_pts = peak_widths[0]
    else:
        min_width_pts = 1
    approx_fwhm = dx * min_width_pts
    sigma_guess = approx_fwhm / 2.3548 if approx_fwhm > 0 else dx

    # Initialize models
    for peak_idx, model_idx in zip(peak_indices.tolist(), model_indicies):
        model = spec["model"][model_idx]

        if model.get("type") in ["GaussianModel", "LorentzianModel", "VoigtModel"]:
            params = {
                "height": float(y[peak_idx]),
                "sigma": float(sigma_guess),
                "center": float(x[peak_idx]),
            }
            if "params" in model:
                model["params"].update(params)
            else:
                model["params"] = params
        else:
            raise NotImplementedError(
                f"Model type '{model.get('type')}' not implemented in update_spec_from_peaks."
            )

    print(f"Peaks count: {len(peak_indices)}")
    return spec, peak_indices

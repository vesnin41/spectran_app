# src/fitting/generate_spec.py
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from scipy import signal

from src.config import CONFIG


@dataclass
class Peak:
    """Detected peak with position, height, and estimated FWHM."""

    index: int
    two_theta: float
    height: float
    fwhm_est: float


def _detect_peaks_find_peaks(
    x: np.ndarray,
    y: np.ndarray,
    width_range: Tuple[int, int] | None = None,
    rel_height_min: float = 0.05,
    prominence: float | None = None,
    prominence_rel: float | None = None,
    distance: int | None = None,
) -> Tuple[np.ndarray, dict]:
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
        return np.array([], dtype=int), {}

    y_max = float(np.max(y))
    if y_max <= 0:
        return np.array([], dtype=int), {}

    height = rel_height_min * y_max if rel_height_min is not None else None
    prom = prominence if prominence is not None else (
        prominence_rel * y_max if prominence_rel is not None else None
    )

    width = None
    if width_range is not None:
        wmin, wmax = width_range
        width = (max(1, int(wmin)), max(1, int(wmax)))

    peaks, properties = signal.find_peaks(
        y,
        height=height,
        prominence=prom,
        width=width,
        distance=distance,
    )

    return peaks.astype(int), properties


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


def detect_peaks(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    width_range: Tuple[int, int] | None = None,
    rel_height_min: float | None = None,
    prominence: float | None = None,
    prominence_rel: float | None = None,
    distance: int | None = None,
) -> List[Peak]:
    """
    Detect peaks on a preprocessed spectrum and estimate FWHM.
    """
    x = np.asarray(two_theta, dtype=float)
    y = np.asarray(intensity, dtype=float)

    if x.size == 0 or y.size == 0 or x.size != y.size:
        return []

    if width_range is not None:
        width_range = (int(width_range[0]), int(width_range[1]))
    rel_height_min = CONFIG.rel_height_min if rel_height_min is None else rel_height_min
    if prominence_rel is None:
        prominence_rel = CONFIG.prominence_rel

    dist_pts = distance
    if dist_pts is None and width_range is not None:
        dist_pts = max(1, width_range[0])

    peaks_idx, properties = _detect_peaks_find_peaks(
        x,
        y,
        width_range=width_range,
        rel_height_min=rel_height_min,
        prominence=prominence,
        prominence_rel=prominence_rel,
        distance=dist_pts,
    )

    if peaks_idx.size == 0:
        return []

    if x.size > 1:
        dx = float((x.max() - x.min()) / (len(x) - 1))
    else:
        dx = 1.0

    if peaks_idx.size:
        widths_result = signal.peak_widths(y, peaks_idx, rel_height=0.5)
        widths = widths_result[0]  # in sample points
    else:
        widths = None
    if widths is None or len(widths) == 0:
        # Use half of min width points as fallback
        widths = np.full_like(peaks_idx, fill_value=(width_range[0] if width_range else 1), dtype=float)

    fwhm_vals = np.asarray(widths, dtype=float) * dx

    return [
        Peak(
            index=int(idx),
            two_theta=float(x[idx]),
            height=float(y[idx]),
            fwhm_est=float(fwhm),
        )
        for idx, fwhm in zip(peaks_idx.tolist(), fwhm_vals.tolist())
    ]


def _guess_sigma_from_fwhm(fwhm: float, fallback: float) -> float:
    return float(fwhm / 2.3548) if fwhm > 0 else fallback


def build_model_from_peaks(
    peaks: List[Peak],
    model_type: str = "GaussianModel",
    two_theta: Iterable[float] | None = None,
    intensity: Iterable[float] | None = None,
    extra_components: int = 0,
    amorph_components: int = 1,
) -> list[dict]:
    """
    Build model definitions using detected peaks + extra/amorph components.
    """
    peaks_sorted = sorted(peaks, key=lambda p: p.height, reverse=True)
    two_theta_arr = np.asarray(list(two_theta) if two_theta is not None else [], dtype=float)
    intensity_arr = np.asarray(list(intensity) if intensity is not None else [], dtype=float)

    tmin = float(np.min(two_theta_arr)) if two_theta_arr.size else 0.0
    tmax = float(np.max(two_theta_arr)) if two_theta_arr.size else 1.0
    y_max = float(np.max(intensity_arr)) if intensity_arr.size else 1.0
    tmid = 0.5 * (tmin + tmax)

    def _fallback_sigma():
        span = max(tmax - tmin, 1.0)
        return span * 0.01

    models = []

    sigma_cryst_values = []
    for pk in peaks_sorted:
        sigma = _guess_sigma_from_fwhm(pk.fwhm_est, _fallback_sigma())
        sigma_cryst_values.append(sigma)
        amplitude = abs(pk.height) * abs(sigma) * np.sqrt(2 * np.pi)
        models.append(
            {
                "type": model_type,
                "params": {
                    "center": pk.two_theta,
                    "sigma": sigma,
                    "height": max(pk.height, 1e-3),
                    "amplitude": amplitude,
                },
                "meta": {"kind": "crystalline"},
            }
        )

    sigma_med = float(np.median(sigma_cryst_values)) if sigma_cryst_values else _fallback_sigma()
    sigma_wide = max(sigma_med * CONFIG.amorph_sigma_scale, _fallback_sigma() * 3.0)

    extra_components = max(extra_components, 0)
    amorph_components = max(amorph_components, 0)
    amorph_to_place = min(amorph_components, extra_components)
    extra_narrow = extra_components - amorph_to_place

    # Amorphous broad components
    for i in range(amorph_to_place):
        center = tmid if peaks_sorted == [] else float(np.mean([p.two_theta for p in peaks_sorted]))
        # Slightly shift if more than one amorphous component
        if amorph_to_place > 1:
            shift = (i - (amorph_to_place - 1) / 2) * sigma_med * 1.5
            center += shift
            center = float(np.clip(center, tmin, tmax))

        height = CONFIG.amorph_height_fraction * y_max if y_max > 0 else 1.0
        amplitude = abs(height) * abs(sigma_wide) * np.sqrt(2 * np.pi)
        models.append(
            {
                "type": model_type,
                "params": {
                    "center": center,
                    "sigma": sigma_wide,
                    "height": height,
                    "amplitude": amplitude,
                },
                "param_hints": {
                    "sigma": {"min": sigma_med * CONFIG.amorph_sigma_min_mult},
                    "height": {"min": 0},
                },
                "meta": {"kind": "amorphous"},
            }
        )

    # Extra narrow/medium components spread across range
    if extra_narrow > 0:
        centers_grid = np.linspace(tmin, tmax, extra_narrow + 2)[1:-1] if extra_narrow > 1 else [tmid]
        for ctr in centers_grid:
            sigma = sigma_med
            height = CONFIG.extra_height_fraction * y_max if y_max > 0 else 1.0
            amplitude = abs(height) * abs(sigma) * np.sqrt(2 * np.pi)
            models.append(
                {
                    "type": model_type,
                    "params": {
                        "center": float(ctr),
                        "sigma": sigma,
                        "height": height,
                        "amplitude": amplitude,
                    },
                    "meta": {"kind": "crystalline"},
                }
            )

    return models


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
    properties = {}

    if method == "find_peaks":
        if isinstance(peak_widths, tuple) and len(peak_widths) == 2:
            width_range = (int(peak_widths[0]), int(peak_widths[1]))
        else:
            width_range = None

        dist_pts = distance
        if dist_pts is None and width_range is not None:
            dist_pts = max(1, width_range[0])

        peaks, properties = _detect_peaks_find_peaks(
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
    widths = properties.get("widths") if method == "find_peaks" else None
    if widths is not None and len(widths) > 0:
        approx_fwhm = np.median(np.asarray(widths, dtype=float)) * dx
    elif isinstance(peak_widths, tuple) and len(peak_widths) == 2:
        approx_fwhm = dx * peak_widths[0]
    else:
        approx_fwhm = dx
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

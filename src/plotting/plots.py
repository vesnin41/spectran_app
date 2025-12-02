"""
Reusable plotting utilities for spectra and fits.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable, Sequence
from itertools import cycle


def plot_spectrum(
    two_theta: Iterable[float],
    intensity: Iterable[float],
    title: str = "",
    label: str | None = None,
    overlay: Sequence[tuple[Iterable[float], Iterable[float], str]] | None = None,
    show: bool = True,
):
    """
    Plot a single spectrum with optional overlay curves.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(two_theta, intensity, linewidth=1.0, label=label or "Спектр")
    if overlay:
        for x, y, lbl in overlay:
            ax.plot(x, y, linewidth=0.8, alpha=0.6, label=lbl)
    ax.set_xlabel("2θ, град", fontsize=12)
    ax.set_ylabel("Интенсивность, отн. ед.", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=10)
    if title:
        ax.set_title(title, fontsize=12)
    if label or overlay:
        ax.legend(fontsize=10)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_with_peaks(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    peak_indices: np.ndarray,
    title: str = "Найденные пики",
    show: bool = True,
):
    """
    Plot spectrum with vertical lines marking peak indices.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(two_theta, intensity, linewidth=1.0)
    for idx in peak_indices:
        ax.axvline(two_theta[idx], color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("2θ, град", fontsize=12)
    ax.set_ylabel("Интенсивность, отн. ед.", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=10)
    if title:
        ax.set_title(title, fontsize=12)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_fit(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    fit_curve: np.ndarray,
    title: str = "Фит",
    show: bool = True,
):
    """
    Plot experimental spectrum and fitted curve.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(two_theta, intensity, label="Эксперимент", linewidth=1.0)
    ax.plot(two_theta, fit_curve, label="Модель", linewidth=1.0)
    ax.set_xlabel("2θ, град", fontsize=12)
    ax.set_ylabel("Интенсивность, отн. ед.", fontsize=12)
    ax.legend(fontsize=10)
    ax.tick_params(axis="both", which="major", labelsize=10)
    if title:
        ax.set_title(title, fontsize=12)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_fit_with_components(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    result,
    spec: dict | None = None,
    title: str = "Фит с компонентами",
    show: bool = True,
):
    """
    Plot experimental spectrum, total fit, and individual components.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(two_theta, intensity, label="Эксперимент", linewidth=1.0)
    ax.plot(two_theta, result.best_fit, label="Суммарный фит", linewidth=1.2, color="C1")

    kind_by_prefix = {}
    if spec is not None and isinstance(spec, dict) and "model" in spec:
        for i, model_def in enumerate(spec["model"]):
            prefix = f"m{i}_"
            kind_by_prefix[prefix] = (model_def.get("meta") or {}).get("kind", "")

    comps = result.eval_components(x=two_theta)
    for prefix, y_comp in comps.items():
        kind = kind_by_prefix.get(prefix, "")
        if kind == "amorphous":
            style = {"linestyle": "-", "linewidth": 1.4, "alpha": 0.9, "color": "C2"}
            label = f"{prefix} (amorph)"
        else:
            style = {"linestyle": "--", "linewidth": 0.8, "alpha": 0.6}
            label = prefix
        ax.plot(two_theta, y_comp, label=label, **style)

    ax.set_xlabel("2θ, град", fontsize=12)
    ax.set_ylabel("Интенсивность, отн. ед.", fontsize=12)
    if title:
        ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_fit_with_phase_markers(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    fit_curve: np.ndarray,
    peak_table,
    title: str = "Фит с фазами",
    max_components: int | None = None,
    show: bool = True,
):
    """
    Plot fit and mark peaks colored by phase_id.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(two_theta, intensity, label="Эксперимент", linewidth=1.0)
    ax.plot(two_theta, fit_curve, label="Модель", linewidth=1.2)

    colors = cycle(["C2", "C3", "C4", "C5", "C6", "C7"])
    phase_colors = {}

    if hasattr(peak_table, "groupby") and "phase_id" in peak_table.columns:
        for phase_id, df_phase in peak_table.groupby("phase_id"):
            if phase_id is None or (isinstance(phase_id, float) and np.isnan(phase_id)):
                continue
            if phase_id not in phase_colors:
                phase_colors[phase_id] = next(colors)
            color = phase_colors[phase_id]
            centers = df_phase["center"].values
            y_interp = np.interp(centers, two_theta, fit_curve)
            ax.scatter(centers, y_interp, marker="*", s=60, color=color, label=str(phase_id), zorder=3)

    ax.set_xlabel("2θ, град", fontsize=12)
    ax.set_ylabel("Интенсивность, отн. ед.", fontsize=12)
    ax.legend(fontsize=10)
    ax.tick_params(axis="both", which="major", labelsize=10)
    if title:
        ax.set_title(title, fontsize=12)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_ref_preview(
    ref_two_theta: Iterable[float],
    ref_intensity: Iterable[float],
    exp_two_theta: Iterable[float] | None = None,
    exp_intensity: Iterable[float] | None = None,
    title: str | None = None,
    show: bool = True,
):
    """
    Preview a reference pattern as vertical sticks; optionally overlay experimental spectrum.
    """
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    if exp_two_theta is not None and exp_intensity is not None:
        ax.plot(exp_two_theta, exp_intensity, label="Эксперимент", linewidth=1.0)
    ax.vlines(ref_two_theta, 0, ref_intensity, colors="red", alpha=0.7, label="CIF")
    ax.set_xlabel("2θ, град")
    ax.set_ylabel("Интенсивность, отн. ед.")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax

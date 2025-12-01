"""
Reusable plotting utilities for spectra and fits.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable, Sequence


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

"""
Helper to plot fits with component curves.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_fit_with_components(two_theta: np.ndarray, intensity: np.ndarray, model_output, max_components: int | None = None):
    """
    Plot experiment, best fit, and individual components (if available).
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(two_theta, intensity, label="Эксперимент", linewidth=1.0)
    ax.plot(two_theta, model_output.best_fit, label="Модель", linewidth=1.0)

    comps = model_output.eval_components(x=two_theta)
    for idx, (name, comp) in enumerate(comps.items()):
        if max_components is not None and idx >= max_components:
            break
        ax.plot(two_theta, comp, linewidth=0.8, alpha=0.6, label=name)

    ax.set_xlabel("2θ, град", fontsize=12)
    ax.set_ylabel("Интенсивность, отн. ед.", fontsize=12)
    ax.legend(fontsize=10)
    ax.tick_params(axis="both", which="major", labelsize=10)
    fig.tight_layout()
    return fig, ax

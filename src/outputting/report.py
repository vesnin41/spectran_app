"""
Shared helpers for reporting/saving fit results and plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.plots import (
    plot_fit,
    plot_fit_with_components,
    plot_fit_with_phase_markers,
    plot_residuals,
    plot_spectrum,
)


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_fit_outputs(
    result_dir: Path,
    spec: dict,
    output,
    peak_table: pd.DataFrame,
    preprocessing_cfg: dict,
    ci: float | None = None,
    fit_cfg: dict | None = None,
) -> None:
    """
    Save plots and CSVs similar to CLI behavior.
    """
    result_dir.mkdir(parents=True, exist_ok=True)

    # Peak table
    peak_table.to_csv(result_dir / "peak_table.csv", index=False)

    # Fit curves CSV
    fit_df = pd.DataFrame(
        {
            "two_theta": spec["x"],
            "intensity": spec["y"],
            "best_fit": output.best_fit,
            "residual": output.residual,
        }
    )
    fit_df.to_csv(result_dir / "fit_curve.csv", index=False)

    # Plots
    fig_fit, _ = plot_fit(spec["x"], spec["y"], output.best_fit, show=False)
    save_fig(fig_fit, result_dir / "fit.png")

    fig_comp, _ = plot_fit_with_components(spec["x"], spec["y"], output, spec=spec, peak_table=peak_table, show=False)
    save_fig(fig_comp, result_dir / "fit_components.png")

    residual_var = float(np.var(output.residual))
    fig_res, _ = plot_residuals(
        spec["x"], output.residual, title=f"Остатки (var={residual_var:.4f})", show=False
    )
    save_fig(fig_res, result_dir / "fit_residuals.png")

    if "phase_id" in peak_table.columns:
        fig_phase, _ = plot_fit_with_phase_markers(spec["x"], spec["y"], output.best_fit, peak_table, show=False)
        save_fig(fig_phase, result_dir / "fit_phases.png")

    # Raw vs processed plot if data present
    if preprocessing_cfg.get("raw_two_theta") is not None:
        fig_raw, _ = plot_spectrum(
            spec["x"],
            spec["y"],
            overlay=[(preprocessing_cfg["raw_two_theta"], preprocessing_cfg["raw_intensity"], "Сырой")],
            label="Обработанный",
            title="Сырой vs обработанный",
            show=False,
        )
        save_fig(fig_raw, result_dir / "raw_vs_processed.png")

    # Summary text
    meta_path = result_dir / "summary.txt"
    with meta_path.open("w", encoding="utf-8") as fh:
        fh.write(f"CI (%): {ci:.2f}\n" if ci is not None else "")
        fh.write("Preprocessing:\n")
        for k, v in preprocessing_cfg.items():
            if k.startswith("raw_"):
                continue
            fh.write(f"  {k}: {v}\n")
        if fit_cfg:
            fh.write("Fitting:\n")
            for k, v in fit_cfg.items():
                fh.write(f"  {k}: {v}\n")


def build_phase_summary_text(phase_results: Iterable, fractions: dict) -> list[str]:
    """
    Human-readable lines for phase summary.
    """
    lines = [
        f"Кристалличность: {fractions.get('x_cryst',0)*100:.1f} %, "
        f"Аморфность: {fractions.get('x_amorph',0)*100:.1f} %"
    ]
    for pr in sorted(phase_results, key=lambda p: p.area_total, reverse=True):
        d_avg = pr.mean_crystallite_size()
        d_str = f"{d_avg:.2f}" if d_avg is not None else "--"
        share = (
            pr.area_total / fractions.get("area_total", 1) * 100 if fractions.get("area_total", 0) else 0.0
        )
        lines.append(
            f"{pr.phase_id}: N={pr.n_peaks}, Σ={pr.area_total:.2f}, D_avg={d_str} нм, доля={share:.1f}%"
        )
    return lines

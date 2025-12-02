"""
Aggregate fitted peaks by phase and estimate area fractions / crystallite size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.utils.peak_metrics import FittedPeak
from src.utils.xrd_geometry import get_crystallite_size_scherrer


@dataclass
class PhaseResult:
    phase_id: str
    peaks: list[FittedPeak]

    @property
    def area_total(self) -> float:
        return float(sum(p.area for p in self.peaks))

    @property
    def n_peaks(self) -> int:
        return len(self.peaks)

    def mean_crystallite_size(
        self,
        wavelength_nm: float = 0.154060,
        k: float = 0.9,
        instrument_fwhm_deg: float | None = None,
    ) -> float | None:
        """
        Area-weighted Scherrer crystallite size across peaks of the phase.
        """
        if not self.peaks:
            return None

        sizes = []
        weights = []
        for pk in self.peaks:
            if pk.fwhm <= 0 or np.isnan(pk.fwhm) or np.isnan(pk.two_theta):
                continue

            beta_deg = pk.fwhm
            if instrument_fwhm_deg is not None and instrument_fwhm_deg > 0:
                beta_obs = np.deg2rad(beta_deg)
                beta_instr = np.deg2rad(instrument_fwhm_deg)
                beta_net = np.sqrt(max(beta_obs**2 - beta_instr**2, 1e-12))
                beta_deg = np.rad2deg(beta_net)

            size = get_crystallite_size_scherrer(
                fwhm_deg=beta_deg,
                center_deg=pk.two_theta,
                wavelength_nm=wavelength_nm,
                k=k,
            )
            if np.isnan(size):
                continue
            sizes.append(size)
            weights.append(max(pk.area, 1e-9))

        if not sizes or not weights:
            return None

        sizes = np.asarray(sizes, dtype=float)
        weights = np.asarray(weights, dtype=float)
        return float(np.average(sizes, weights=weights))


def aggregate_phase_results(peaks: Iterable[FittedPeak]) -> list[PhaseResult]:
    by_phase: dict[str, list[FittedPeak]] = {}
    for pk in peaks:
        pid = pk.phase_id
        if pid is None:
            continue
        by_phase.setdefault(str(pid), []).append(pk)
    return [PhaseResult(phase_id=k, peaks=v) for k, v in by_phase.items()]


def compute_area_fractions(
    peaks: Iterable[FittedPeak], amorphous_phase_id: str = "amorphous"
) -> dict:
    peaks_list = list(peaks)
    area_total = float(sum(max(p.area, 0.0) for p in peaks_list))
    if area_total <= 0:
        return {
            "area_total": 0.0,
            "area_cryst": 0.0,
            "area_amorph": 0.0,
            "x_cryst": 0.0,
            "x_amorph": 0.0,
        }

    area_amorph = float(
        sum(
            max(p.area, 0.0)
            for p in peaks_list
            if p.is_amorphous or (p.phase_id is not None and str(p.phase_id) == amorphous_phase_id)
        )
    )
    area_cryst = max(area_total - area_amorph, 0.0)

    return {
        "area_total": area_total,
        "area_cryst": area_cryst,
        "area_amorph": area_amorph,
        "x_cryst": area_cryst / area_total if area_total else 0.0,
        "x_amorph": area_amorph / area_total if area_total else 0.0,
    }

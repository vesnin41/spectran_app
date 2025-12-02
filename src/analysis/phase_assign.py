"""
Assign fitted peaks to phases using existing search-match utilities.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.peak_metrics import FittedPeak
from src.utils.search_match import annotate_peaks_with_phases


def assign_phases_to_peaks(
    fitted_peaks: list[FittedPeak],
    ref_db: Iterable[pd.DataFrame],
    max_delta_two_theta: float = 0.2,
    amorphous_phase_id: str = "amorphous",
) -> None:
    """
    Mutates fitted_peaks in-place, filling phase_id where matches are found.
    """
    if not fitted_peaks:
        return

    # Amorphous peaks get fixed phase id
    for pk in fitted_peaks:
        if pk.is_amorphous:
            pk.phase_id = amorphous_phase_id

    # If no references provided, nothing else to do
    ref_db_list = list(ref_db)
    if not ref_db_list:
        return

    exp_rows = []
    idx_map: list[int] = []
    for i, pk in enumerate(fitted_peaks):
        if pk.is_amorphous:
            continue
        if np.isnan(pk.two_theta):
            continue
        exp_rows.append({"center": pk.two_theta, "height": pk.height})
        idx_map.append(i)

    if not exp_rows:
        return

    exp_df = pd.DataFrame(exp_rows)
    matches = annotate_peaks_with_phases(exp_df, ref_db_list, delta_2theta_max=max_delta_two_theta)
    if matches.empty:
        return

    for _, row in matches.iterrows():
        exp_index = int(row["exp_index"])
        if 0 <= exp_index < len(idx_map):
            pk_idx = idx_map[exp_index]
            fitted_peaks[pk_idx].phase_id = row["phase_id"]

# src/utils/phase_match.py

from __future__ import annotations

from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd

from src.utils.spectrum_math import normalize_data


def _euclidean_distance(x0, y0, xf, yf) -> float:
    """Евклидово расстояние между точками (x0, y0) и (xf, yf)."""
    return np.sqrt((x0 - xf) ** 2 + (y0 - yf) ** 2)


# ==========================
#  1. ОДИН ПИК – ОДНА ФАЗА
# ==========================

def match_peak_to_single_phase(
    peak_row: pd.Series,
    phase_df: pd.DataFrame,
    delta_2theta: float = 0.2,
    use_intensity_in_distance: bool = True,
) -> pd.Series:
    """
    Сопоставление ОДНОГО экспериментального пика с ОДНОЙ эталонной фазой.

    Parameters
    ----------
    peak_row : pd.Series
        Ожидает поля:
            - 'center' : положение пика (2θ, град)
            - 'height' : интенсивность (после нормализации)
    phase_df : pd.DataFrame
        Эталонные пики ОДНОЙ фазы, столбцы:
            - 'two_theta'
            - 'intensity'
    delta_2theta : float
        Максимальное расстояние по 2θ (в градусах), в котором ищутся совпадения.
    use_intensity_in_distance : bool
        Если True: используем евклидово расстояние в пространстве (2θ, I).
        Если False: расстояние только по 2θ.

    Returns
    -------
    pd.Series с полями:
        - distance : минимальное расстояние (float или NaN)
        - is_match : bool, есть совпадение или нет
    """
    x0 = float(peak_row["center"])
    y0 = float(peak_row["height"])

    if phase_df.empty:
        return pd.Series({"distance": np.nan, "is_match": False})

    mask = np.abs(phase_df["two_theta"] - x0) <= delta_2theta
    candidates = phase_df.loc[mask]

    if candidates.empty:
        return pd.Series({"distance": np.nan, "is_match": False})

    if use_intensity_in_distance:
        distances = np.sqrt(
            (candidates["two_theta"] - x0) ** 2
            + (candidates["intensity"] - y0) ** 2
        )
    else:
        distances = np.abs(candidates["two_theta"] - x0)

    idx_min = distances.idxmin()
    distance_min = float(distances.loc[idx_min])

    return pd.Series({"distance": distance_min, "is_match": True})


def match_peaks_to_single_phase(
    peaks_df: pd.DataFrame,
    phase_df: pd.DataFrame,
    phase_label: Optional[str] = None,
    delta_2theta: float = 0.2,
    use_intensity_in_distance: bool = True,
) -> pd.DataFrame:
    """
    Сопоставление ВСЕХ экспериментальных пиков с ОДНОЙ фазой.

    Результат: к peaks_df добавляются 2 столбца:
        - dist_{phase_label}
        - is_{phase_label}

    Если phase_label не задан, используются 'dist_phase' и 'is_phase'.
    """
    if phase_label is None:
        dist_col = "dist_phase"
        is_col = "is_phase"
    else:
        dist_col = f"dist_{phase_label}"
        is_col = f"is_{phase_label}"

    res = peaks_df.apply(
        match_peak_to_single_phase,
        axis=1,
        phase_df=phase_df,
        delta_2theta=delta_2theta,
        use_intensity_in_distance=use_intensity_in_distance,
    )

    peaks_df = peaks_df.copy()
    peaks_df[dist_col] = res["distance"]
    peaks_df[is_col] = res["is_match"]

    return peaks_df


# ==========================
#  2. ОДИН ПИК – ВСЕ ФАЗЫ
# ==========================

def assign_phase_by_nearest_peak(
    peak_row: pd.Series,
    all_phases_df: pd.DataFrame,
    delta_2theta: float = 0.2,
    use_intensity_in_distance: bool = True,
) -> pd.Series:
    """
    Сопоставление ОДНОГО пика со ВСЕМИ фазами сразу
    и выбор ближайшего по расстоянию эталонного пика.

    all_phases_df должно содержать:
        - 'two_theta'
        - 'intensity'
        - 'id' : имя/код фазы (например, 'HA', 'CA', 'TCP_alpha'...)

    Возвращает:
        - distance : минимальное расстояние до эталонного пика
        - phase_id : id фазы (строка) или NaN, если совпадений нет
    """
    x0 = float(peak_row["center"])
    y0 = float(peak_row["height"])

    if all_phases_df.empty:
        return pd.Series({"distance": np.nan, "phase_id": np.nan})

    mask = np.abs(all_phases_df["two_theta"] - x0) <= delta_2theta
    candidates = all_phases_df.loc[mask]

    if candidates.empty:
        return pd.Series({"distance": np.nan, "phase_id": np.nan})

    if use_intensity_in_distance:
        distances = np.sqrt(
            (candidates["two_theta"] - x0) ** 2
            + (candidates["intensity"] - y0) ** 2
        )
    else:
        distances = np.abs(candidates["two_theta"] - x0)

    idx_min = distances.idxmin()
    distance_min = float(distances.loc[idx_min])
    phase_id = candidates.loc[idx_min, "id"]

    return pd.Series({"distance": distance_min, "phase_id": phase_id})


def assign_phases_for_peaks(
    peaks_df: pd.DataFrame,
    all_phases_df: pd.DataFrame,
    delta_2theta: float = 0.2,
    use_intensity_in_distance: bool = True,
) -> pd.DataFrame:
    """
    Назначить фазу каждому экспериментальному пику —
    по ближайшему эталонному пику из общей таблицы all_phases_df.

    all_phases_df — это конкатенация всех фаз,
    где столбец 'id' содержит имя фазы ('HA', 'CA', 'TCP_alpha', ...).

    Результат: к peaks_df добавляются столбцы:
        - 'dist'  : минимальное расстояние до базы
        - 'phase' : строковое обозначение фазы
    """
    res = peaks_df.apply(
        assign_phase_by_nearest_peak,
        axis=1,
        all_phases_df=all_phases_df,
        delta_2theta=delta_2theta,
        use_intensity_in_distance=use_intensity_in_distance,
    )

    peaks_df = peaks_df.copy()
    peaks_df["dist"] = res["distance"]
    peaks_df["phase"] = res["phase_id"]

    return peaks_df


# ==========================
#  3. FoM (Figure of Merit) Search–Match
# ==========================

def compute_fom(
    exp_peaks: pd.DataFrame,
    ref_peaks: pd.DataFrame,
    delta_2theta: float = 0.2,
    exp_center_col: str = "center",
    exp_intensity_col: str = "height",
    ref_center_col: str = "two_theta",
    ref_intensity_col: str = "intensity",
    log_components: bool = False,
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    """
    Рассчитать FoM (Figure of Merit) для пары эксперимент / эталон по алгоритму Search–Match.

    Идея:
    - сопоставить пики один-к-одному (один экспериментальный пик ↔ один пик базы)
      в пределах окна ±delta_2theta;
    - по совпавшим парам пиков вычислить:
        F_theta : вклад от расхождения 2θ,
        F_I     : вклад от расхождения интенсивностей,
        F_exp   : доля совпадающих пиков от числа экспериментальных,
        F_ref   : доля совпадающих пиков от числа эталонных;
    - итоговый FoM = (F_theta + F_I + F_exp + F_ref) / 4.

    Параметры
    ---------
    exp_peaks : DataFrame
        Экспериментальные пики, должны содержать:
            - exp_center_col  (по умолчанию 'center')
            - exp_intensity_col (по умолчанию 'height')
    ref_peaks : DataFrame
        Эталонные пики одной фазы, должны содержать:
            - ref_center_col  (по умолчанию 'two_theta')
            - ref_intensity_col (по умолчанию 'intensity')
    delta_2theta : float
        Максимальный допуск по 2θ (град).
    exp_center_col, exp_intensity_col, ref_center_col, ref_intensity_col : str
        Имена столбцов с центрами пиков и интенсивностями.

    Returns
    -------
    fom : float
        Итоговая метрика соответствия в диапазоне [0, 1].
    components : dict
        Отдельные вклады:
            {
            'F_theta': ...,
            'F_I': ...,
            'F_exp': ...,
            'F_ref': ...,
            'N_match': ...,
            'N_exp': ...,
            'N_ref': ...
            }
    matches_df : DataFrame
        Таблица совпавших пиков:
            ['center_exp', 'intensity_exp',
             'center_ref', 'intensity_ref',
             'dtheta', 'dI']
    """
    # Копим локальные копии
    exp = exp_peaks[[exp_center_col, exp_intensity_col]].dropna().copy()
    ref = ref_peaks[[ref_center_col, ref_intensity_col]].dropna().copy()

    exp.columns = ["center_exp", "intensity_exp"]
    ref.columns = ["center_ref", "intensity_ref"]

    N_exp = len(exp)
    N_ref = len(ref)

    if N_exp == 0 or N_ref == 0:
        components = dict(
            F_theta=0.0,
            F_I=0.0,
            F_exp=0.0,
            F_ref=0.0,
            N_match=0,
            N_exp=N_exp,
            N_ref=N_ref,
        )
        return 0.0, components, pd.DataFrame(columns=[
            "center_exp", "intensity_exp",
            "center_ref", "intensity_ref",
            "dtheta", "dI"
        ])

    # Нормируем интенсивности в [0, 1]
    exp["intensity_exp_norm"] = normalize_data(exp["intensity_exp"].values)
    ref["intensity_ref_norm"] = normalize_data(ref["intensity_ref"].values)

    # Greedy-сопоставление пиков: идём по эксп. пикам в порядке убывания высоты
    exp_sorted = exp.sort_values("intensity_exp_norm", ascending=False).reset_index(drop=True)
    ref_available = ref.copy()
    used_ref_indices: List[int] = []

    matches = []

    for _, row_exp in exp_sorted.iterrows():
        x0 = float(row_exp["center_exp"])
        y0 = float(row_exp["intensity_exp_norm"])

        # кандидаты по 2θ
        mask = np.abs(ref_available["center_ref"] - x0) <= delta_2theta
        candidates = ref_available.loc[mask]

        if candidates.empty:
            continue

        # выбираем ближайший по 2θ
        dtheta = np.abs(candidates["center_ref"] - x0)
        idx_local = dtheta.idxmin()
        row_ref = candidates.loc[idx_local]

        # фиксируем совпадение
        matches.append(
            dict(
                center_exp=x0,
                intensity_exp=row_exp["intensity_exp"],
                center_ref=float(row_ref["center_ref"]),
                intensity_ref=row_ref["intensity_ref"],
                dtheta=float(abs(row_ref["center_ref"] - x0)),
                dI=float(abs(row_ref["intensity_ref_norm"] - y0)),
            )
        )

        # этот эталонный пик больше нельзя использовать
        ref_available = ref_available.drop(index=idx_local)

    if not matches:
        components = dict(
            F_theta=0.0,
            F_I=0.0,
            F_exp=0.0,
            F_ref=0.0,
            N_match=0,
            N_exp=N_exp,
            N_ref=N_ref,
        )
        return 0.0, components, pd.DataFrame(columns=[
            "center_exp", "intensity_exp",
            "center_ref", "intensity_ref",
            "dtheta", "dI"
        ])

    matches_df = pd.DataFrame(matches)
    N_match = len(matches_df)

    # --- Вклады ---

    # 1) вклад от расхождения по 2θ
    dtheta_max = float(delta_2theta)
    F_theta = 1.0 - (matches_df["dtheta"] / dtheta_max).mean()
    F_theta = float(np.clip(F_theta, 0.0, 1.0))

    # 2) вклад от расхождения по интенсивности (нормированной)
    F_I = 1.0 - matches_df["dI"].mean()
    F_I = float(np.clip(F_I, 0.0, 1.0))

    # 3) доля совпадающих пиков от числа экспериментальных
    F_exp = float(N_match / N_exp)

    # 4) доля совпадающих пиков от числа эталонных
    F_ref = float(N_match / N_ref)

    # Итоговая метрика
    fom = float(np.clip((F_theta + F_I + F_exp + F_ref) / 4.0, 0.0, 1.0))

    components = dict(
        F_theta=F_theta,
        F_I=F_I,
        F_exp=F_exp,
        F_ref=F_ref,
        N_match=N_match,
        N_exp=N_exp,
        N_ref=N_ref,
    )

    if log_components:
        print(f"FoM components: {components}")

    return fom, components, matches_df

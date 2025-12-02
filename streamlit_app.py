"""
Streamlit interface for XRD/Raman/FTIR spectrum analysis.
"""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.analysis.phase_assign import assign_phases_to_peaks
from src.analysis.phase_summary import aggregate_phase_results, compute_area_fractions
from src.config import CONFIG
from src.fitting import fit_model, generate_spec
from src.plotting.plots import (
    plot_fit_with_components,
    plot_ref_preview,
    plot_residuals,
    plot_spectrum,
    plot_with_peaks,
)
from src.reading.data_from import spectrum_from_csv
from src.utils.peak_metrics import (
    compute_ci,
    fitted_peaks_from_table,
    reclassify_amorphous_by_width,
)
from src.utils.preprocessing import (
    apply_savgol,
    normalize_intensity,
    subtract_baseline,
    trim_2theta,
)
from src.utils.search_match import build_ref_pattern_from_cif, search_match_all_phases
from src.utils.xrd_geometry import get_crystallite_size_scherrer, get_d_hkl


st.set_page_config(page_title="Spectrum Analyzer", layout="wide")
st.title("XRD/FTIR/Raman Spectrum Analyzer")


DATA_DIR = Path("data")
CIF_DIR = Path("CIF")


def list_data_files() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return sorted([p.name for p in DATA_DIR.glob("*.csv")])


def load_spectrum(file_name: str) -> pd.DataFrame:
    path = DATA_DIR / file_name
    return spectrum_from_csv(str(path))


def download_csv(df: pd.DataFrame, name: str) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(f"Скачать {name} (CSV)", data=csv_bytes, file_name=f"{name}.csv", mime="text/csv")


with st.sidebar:
    st.header("Загрузка данных")
    uploaded_file = st.file_uploader("Загрузите CSV-файл спектра", type="csv")
    if uploaded_file is not None:
        save_path = DATA_DIR / uploaded_file.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Файл сохранен: {save_path}")

    files = list_data_files()
    spectrum_file = st.selectbox("Выберите файл из data/", files) if files else None

    st.header("Выбор CIF для search-match")
    available_cif = sorted([p.name for p in CIF_DIR.glob("*.cif")]) if CIF_DIR.exists() else []
    chosen_cif = st.multiselect("Файлы CIF", available_cif)

if not spectrum_file:
    st.info("Загрузите или выберите файл спектра в сайдбаре.")
    st.stop()

df_raw = load_spectrum(spectrum_file)
st.subheader("Предпросмотр данных")
st.write(df_raw.head())

# Preprocessing controls
st.sidebar.header("Предобработка")
theta_min = st.sidebar.number_input("Минимальный 2θ", value=float(df_raw["two_theta"].min()))
theta_max = st.sidebar.number_input("Максимальный 2θ", value=float(df_raw["two_theta"].max()))
window = st.sidebar.slider("Окно Savitzky–Golay", min_value=5, max_value=101, step=2, value=CONFIG.savgol_window)
poly = st.sidebar.slider("Степень полинома", min_value=1, max_value=5, value=CONFIG.savgol_poly)
use_baseline = st.sidebar.checkbox("Вычитать baseline", value=True)
use_norm = st.sidebar.checkbox("Нормировать 0–100%", value=True)

df_proc = trim_2theta(df_raw, interval=(theta_min, theta_max))
df_proc = df_proc.copy()
df_proc["intensity"] = apply_savgol(df_proc["intensity"], window_length=window, polyorder=poly)
if use_baseline:
    df_proc["intensity"] = subtract_baseline(df_proc["two_theta"], df_proc["intensity"])
if use_norm:
    df_proc["intensity"] = normalize_intensity(df_proc["intensity"]) * 100

st.subheader("Сырой vs предобработанный")
fig_proc, _ = plot_spectrum(
    df_proc["two_theta"],
    df_proc["intensity"],
    overlay=[(df_raw["two_theta"], df_raw["intensity"], "Сырой")],
    label="Предобработанный",
    title="Сырой vs предобработанный",
    show=False,
)
st.pyplot(fig_proc)

# Optional: overlay selected CIF on processed spectrum
if chosen_cif:
    ref_tt: list[float] = []
    ref_int: list[float] = []
    for name in chosen_cif:
        try:
            ref_df = build_ref_pattern_from_cif(str(CIF_DIR / name), interval_2theta=(theta_min, theta_max))
            ref_tt.extend(ref_df["two_theta"].tolist())
            ref_int.extend(ref_df["intensity"].tolist())
        except Exception as exc:
            st.warning(f"Не удалось построить CIF {name}: {exc}")
    if ref_tt:
        st.subheader("Предобработанный + выбранные CIF")
        fig_cif, _ = plot_ref_preview(
            ref_two_theta=ref_tt,
            ref_intensity=ref_int,
            exp_two_theta=df_proc["two_theta"],
            exp_intensity=df_proc["intensity"],
            title="Предобработанный + CIF",
            show=False,
        )
        st.pyplot(fig_cif)

# Peak detection
st.subheader("Поиск пиков")
peak_width_min = st.number_input(
    "Минимальная ширина (точек)", min_value=1, value=getattr(CONFIG, "peak_width_min", 7)
)
peak_width_max = st.number_input(
    "Максимальная ширина (точек)",
    min_value=peak_width_min + 1,
    value=getattr(CONFIG, "peak_width_max", 30),
)
if st.button("Найти пики"):
    peaks = generate_spec.detect_peaks(
        df_proc["two_theta"].values,
        df_proc["intensity"].values,
        width_range=(peak_width_min, peak_width_max),
        rel_height_min=CONFIG.rel_height_min,
        prominence_rel=CONFIG.prominence_rel,
    )
    st.session_state["peaks"] = peaks
    st.success(f"Найдено пиков: {len(peaks)}")

peaks: List[generate_spec.Peak] = st.session_state.get("peaks", [])
if peaks:
    idxs = np.array([p.index for p in peaks], dtype=int)
    fig_peaks, _ = plot_with_peaks(df_proc["two_theta"].values, df_proc["intensity"].values, idxs, show=False)
    st.pyplot(fig_peaks)

# Fitting controls
st.subheader("Фитирование")
model_type = st.selectbox("Тип модели", ["GaussianModel", "LorentzianModel", "VoigtModel"])
extra_components = st.number_input("Дополнительные компоненты", min_value=0, value=4)
amorph_components = CONFIG.amorph_components_default

if st.button("Запустить фитинг"):
    if not peaks:
        peaks = generate_spec.detect_peaks(
            df_proc["two_theta"].values,
            df_proc["intensity"].values,
            width_range=(peak_width_min, peak_width_max),
            rel_height_min=CONFIG.rel_height_min,
            prominence_rel=CONFIG.prominence_rel,
        )
    models_arr = generate_spec.build_model_from_peaks(
        peaks,
        model_type=model_type,
        two_theta=df_proc["two_theta"].values,
        intensity=df_proc["intensity"].values,
        extra_components=extra_components,
        amorph_components=amorph_components,
    )
    spec = generate_spec.default_spec(df_proc, models_arr)
    model = fit_model.Model(spec)
    output = model.fit()
    residual_var = float(np.var(output.residual))

    peak_table = model.print_best_values()
    peak_table["d_hkl"] = peak_table["center"].apply(get_d_hkl)
    peak_table["crystal_size_nm"] = peak_table.apply(
        lambda row: get_crystallite_size_scherrer(row["fwhm"], row["center"]), axis=1
    )
    peak_table, fwhm_thr = reclassify_amorphous_by_width(peak_table)

    fitted_peaks = fitted_peaks_from_table(peak_table)

    # Phase selection from CIF dir (sidebar choice)
    ref_db = []
    for name in chosen_cif:
        try:
            ref_db.append(build_ref_pattern_from_cif(str(CIF_DIR / name), interval_2theta=(theta_min, theta_max)))
        except Exception as exc:
            st.warning(f"Не удалось загрузить {name}: {exc}")

    if ref_db:
        assign_phases_to_peaks(fitted_peaks, ref_db=ref_db, max_delta_two_theta=CONFIG.delta_2theta_max)
        search_results = search_match_all_phases(peak_table, ref_db=ref_db, delta_2theta_max=CONFIG.delta_2theta_max)
    else:
        search_results = pd.DataFrame()

    phase_results = aggregate_phase_results(fitted_peaks)
    fractions = compute_area_fractions(fitted_peaks)
    ci = compute_ci(
        df_proc["two_theta"].values,
        df_proc["intensity"].values,
        peak_table,
        fwhm_min=CONFIG.ci_fwhm_min,
        fwhm_max=CONFIG.ci_fwhm_max,
        rel_height_min=CONFIG.rel_height_min,
    )

    st.success(f"FWHM порог аморфных: {fwhm_thr:.2f}° 2θ")
    st.write(f"Индекс кристалличности CI: {ci:.1f} %")
    st.write(f"Дисперсия остатков: {residual_var:.4f}")
    if chosen_cif:
        st.write(f"Выбранные CIF: {', '.join(chosen_cif)}")

    st.subheader("Таблица пиков")
    st.dataframe(peak_table)
    download_csv(peak_table, "peaks")

    fig_fit, _ = plot_fit_with_components(spec["x"], spec["y"], output, spec=spec, peak_table=peak_table, show=False)
    st.pyplot(fig_fit)
    fig_res, _ = plot_residuals(spec["x"], output.residual, title=f"Остатки (var={residual_var:.4f})", show=False)
    st.pyplot(fig_res)

    fit_curve = pd.DataFrame(
        {"two_theta": spec["x"], "intensity": spec["y"], "best_fit": output.best_fit, "residual": output.residual}
    )
    download_csv(fit_curve, "fit_curve")

    if not search_results.empty:
        st.subheader("Search-match")
        st.dataframe(search_results)
        download_csv(search_results, "search_match")

    st.subheader("Фазовая сводка")
    st.write(f"Кристалличность: {fractions['x_cryst']*100:.1f} %, Аморфность: {fractions['x_amorph']*100:.1f} %")
    for pr in sorted(phase_results, key=lambda p: p.area_total, reverse=True):
        d_avg = pr.mean_crystallite_size()
        d_str = f"{d_avg:.2f}" if d_avg is not None else "--"
        st.write(
            f"{pr.phase_id}: N={pr.n_peaks}, Σплощадь={pr.area_total:.2f}, D_avg={d_str} нм, "
            f"доля={pr.area_total/ fractions['area_total']*100 if fractions['area_total'] else 0:.1f}%"
        )

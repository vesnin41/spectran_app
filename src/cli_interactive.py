"""
Interactive CLI around the existing XRD processing pipeline.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import CONFIG

from src.analysis.phase_assign import assign_phases_to_peaks
from src.analysis.phase_summary import aggregate_phase_results, compute_area_fractions

from src.fitting import fit_model, generate_spec
from src.reading.data_from import spectrum_from_csv
from src.utils.peak_metrics import compute_ci, fitted_peaks_from_table
from src.utils.preprocessing import (
    apply_savgol,
    normalize_intensity,
    subtract_baseline,
    trim_2theta,
)
from src.utils.xrd_geometry import get_crystallite_size_scherrer, get_d_hkl
from src.plotting.plots import (
    plot_spectrum,
    plot_with_peaks,
    plot_fit,
    plot_fit_with_phase_markers,
    plot_fit_with_components,
    plot_ref_preview,
)
from src.utils.peak_metrics import select_crystalline_peaks
from src.utils.search_match import (
    annotate_peaks_with_phases,
    annotate_peaks_with_best_phase,
    build_ref_pattern_from_cif,
    search_match_all_phases,
)


def ask_int(prompt: str, default: int | None = None) -> int:
    """Prompt for integer with optional default."""
    while True:
        s = input(prompt).strip()
        if not s and default is not None:
            return default
        try:
            return int(s)
        except ValueError:
            print("Введите целое число или оставьте пустым для значения по умолчанию.")


def ask_float(prompt: str, default: float | None = None) -> float:
    """Prompt for float with optional default."""
    while True:
        s = input(prompt).strip()
        if not s and default is not None:
            return default
        try:
            return float(s)
        except ValueError:
            print("Введите число или оставьте пустым для значения по умолчанию.")


def choose_file_interactive() -> str:
    """Offer files from ./data or ask for manual path."""
    print("\nВыбор файла с дифрактограммой.")
    data_dir = Path("data")
    candidates: list[Path] = []
    if data_dir.is_dir():
        candidates = sorted(p for p in data_dir.glob("*.csv"))
        if candidates:
            print("Найдены файлы в каталоге data/:")
            for i, p in enumerate(candidates, 1):
                print(f"[{i}] {p}")
    print("[0] Ввести путь вручную")

    choice = ask_int("Ваш выбор: ", default=0)
    if choice == 0 or not candidates:
        s = input("Введите путь к файлу: ").strip()
        return s
    idx = max(1, min(choice, len(candidates)))
    return str(candidates[idx - 1])


def save_results(
    spectr_path: str,
    spec: dict,
    output,
    peak_table: pd.DataFrame,
    ci: float,
    preprocessing_cfg: dict,
    fit_cfg: dict | None = None,
) -> None:
    """Save fit results and figures to results/ directory."""
    sample_name = Path(spectr_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path("results") / f"{sample_name}_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    peak_table.to_csv(result_dir / "peak_table.csv", index=False)
    # Save experimental and fitted curve
    fit_df = pd.DataFrame({"two_theta": spec["x"], "intensity": spec["y"], "best_fit": output.best_fit})
    fit_df.to_csv(result_dir / "fit_curve.csv", index=False)

    # Fit plot
    fig, _ = plot_fit(spec["x"], spec["y"], output.best_fit, show=False)
    fig.savefig(result_dir / "fit.png", dpi=300)
    plt.close(fig)

    # Components plot
    fig_comp, _ = plot_fit_with_components(spec["x"], spec["y"], output, spec=spec, show=False)
    fig_comp.savefig(result_dir / "fit_components.png", dpi=300)
    plt.close(fig_comp)

    # Phase markers plot (if phase_id present)
    if "phase_id" in peak_table.columns:
        fig_phase, _ = plot_fit_with_phase_markers(spec["x"], spec["y"], output.best_fit, peak_table, show=False)
        fig_phase.savefig(result_dir / "fit_phases.png", dpi=300)
        plt.close(fig_phase)

    # Raw vs processed if available in cfg
    if preprocessing_cfg.get("raw_two_theta") is not None:
        fig_raw, _ = plot_spectrum(
            spec["x"],
            spec["y"],
            overlay=[
                (preprocessing_cfg["raw_two_theta"], preprocessing_cfg["raw_intensity"], "Сырой"),
            ],
            label="Обработанный",
            title="Сырой vs обработанный",
            show=False,
        )
        fig_raw.savefig(result_dir / "raw_vs_processed.png", dpi=300)
        plt.close(fig_raw)

    # Save meta
    meta_path = result_dir / "summary.txt"
    with meta_path.open("w", encoding="utf-8") as fh:
        fh.write(f"Sample: {sample_name}\n")
        fh.write(f"File: {spectr_path}\n")
        fh.write(f"CI (%): {ci:.2f}\n")
        fh.write("Preprocessing:\n")
        for k, v in preprocessing_cfg.items():
            if k.startswith("raw_"):
                continue
            fh.write(f"  {k}: {v}\n")
        if fit_cfg:
            fh.write("Fitting:\n")
            for k, v in fit_cfg.items():
                fh.write(f"  {k}: {v}\n")

    print(f"Результаты сохранены в: {result_dir}")


def select_phases_menu(
    phases_dir: str = "CIF",
    exp_two_theta: np.ndarray | None = None,
    exp_intensity: np.ndarray | None = None,
    interval_2theta: tuple[float, float] = (20.0, 60.0),
) -> list[pd.DataFrame]:
    """
    Interactive selection of CIF phases with preview plots.
    """
    phase_path = Path(phases_dir)
    cif_files = sorted(phase_path.glob("*.cif"))
    if not cif_files:
        print(f"[WARN] В каталоге {phases_dir} нет CIF-файлов.")
        return []

    print("\nВыбор фаз (CIF):")
    for i, p in enumerate(cif_files, 1):
        print(f"[{i}] {p.name}")
    print("[0] Закончить выбор")

    ref_db: list[pd.DataFrame] = []

    while True:
        choice = input("Выберите номер CIF для просмотра/добавления (0 — закончить): ").strip()
        if choice == "0":
            break
        if not choice.isdigit():
            continue
        idx = int(choice)
        if not (1 <= idx <= len(cif_files)):
            continue

        cif_path = cif_files[idx - 1]
        print(f"\nЗагружаю {cif_path.name}...")

        try:
            ref_df = build_ref_pattern_from_cif(str(cif_path), interval_2theta=interval_2theta)
        except Exception as exc:
            print(f"[WARN] Не удалось разобрать {cif_path.name}: {exc}")
            continue

        # Показать формулу, если возможно
        try:
            from pymatgen.core import Structure

            struct = Structure.from_file(str(cif_path))
            print("Фаза:", struct.composition.reduced_formula)
        except Exception:
            pass

        # Plot reference pattern (vertical sticks) and optional experimental spectrum
        plot_ref_preview(
            ref_df["two_theta"],
            ref_df["intensity"],
            exp_two_theta=exp_two_theta,
            exp_intensity=exp_intensity,
            title=cif_path.name,
        )

        yn = input("Добавить эту фазу в анализ? [Y/n]: ").strip().lower()
        if yn in ("", "y", "yes"):
            ref_db.append(ref_df)
            print(f"Фаза {cif_path.name} добавлена.")

    return ref_db


def interactive_session() -> None:
    """Run interactive dialogue."""
    print("=== XRD Explorer: интерактивный режим ===")

    while True:
        spectr_path = choose_file_interactive()
        if not os.path.isfile(spectr_path):
            print(f"Файл не найден: {spectr_path}")
            continue

        df_raw = spectrum_from_csv(spectr_path)
        if df_raw.empty:
            print("Пустой или некорректный файл, выберите другой.")
            continue

        print("Показ сырого спектра...")
        plot_spectrum(df_raw["two_theta"], df_raw["intensity"], title="Исходная дифрактограмма")

        action = input("Продолжить с этим файлом? [y]/n или выбрать другой (d): ").strip().lower()
        if action == "n":
            exit_choice = input("Выйти? [y/n]: ").strip().lower()
            if exit_choice in ("y", "yes"):
                return
            continue
        if action == "d":
            continue

        # Range + preprocessing loop
        tmin, tmax = 20.0, 60.0
        while True:
            tmin = ask_float(f"Минимальный 2θ (по умолчанию {CONFIG.theta_min}): ", default=CONFIG.theta_min)
            tmax = ask_float(f"Максимальный 2θ (по умолчанию {CONFIG.theta_max}): ", default=CONFIG.theta_max)
            if tmax <= tmin:
                print("Максимум должен быть больше минимума.")
                continue

            df_proc = trim_2theta(df_raw, interval=(tmin, tmax))

            # Savitzky–Golay options
            sg_choice = input(
                f"Применить сглаживание Савицкого–Голея? [1]Да ({CONFIG.savgol_window}/{CONFIG.savgol_poly}) [2]Да, вручную [0]Нет (Enter=1): "
            ).strip()
            if sg_choice in ("", "1"):
                wl, po = CONFIG.savgol_window, CONFIG.savgol_poly
            elif sg_choice == "2":
                wl = ask_int(
                    f"Окно сглаживания (нечетное, по умолчанию {CONFIG.savgol_window}): ",
                    default=CONFIG.savgol_window,
                )
                po = ask_int(
                    f"Порядок полинома (по умолчанию {CONFIG.savgol_poly}): ",
                    default=CONFIG.savgol_poly,
                )
            else:
                wl, po = None, None
            if wl and po:
                df_proc["intensity"] = apply_savgol(
                    df_proc["intensity"].values, window_length=wl, polyorder=po
                )

            # Baseline
            use_bl = input("Вычитать фон (baseline)? [Y/n]: ").strip().lower()
            if use_bl in ("", "y", "yes"):
                df_proc["intensity"] = subtract_baseline(
                    df_proc["two_theta"].values, df_proc["intensity"].values
                )

            # Normalization
            use_norm = input("Нормировать интенсивность к 0–100%? [Y/n]: ").strip().lower()
            normalized = False
            if use_norm in ("", "y", "yes"):
                df_proc["intensity"] = normalize_intensity(df_proc["intensity"].values) * 100
                normalized = True

            plot_spectrum(
                df_proc["two_theta"],
                df_proc["intensity"],
                title="Предобработанная дифрактограмма",
                overlay=[
                    (df_raw["two_theta"], df_raw["intensity"], "Сырой сигнал"),
                ],
            )

            choice = input(
                "Устроила предобработка? [1] Да  [2] Повторить  [3] Изменить диапазон  [0] Выход: "
            ).strip()
            if choice == "1" or choice == "":
                break
            if choice == "0":
                return
            if choice == "3":
                continue
            # choice == "2": repeat preprocessing with same range
        preprocessing_cfg = {
            "interval": (tmin, tmax),
            "savgol": {"window": wl, "polyorder": po} if wl and po else None,
            "baseline": use_bl in ("", "y", "yes"),
            "normalized_0_100": normalized,
            "raw_two_theta": df_raw["two_theta"],
            "raw_intensity": df_raw["intensity"],
        }

        # Model selection and peak search loop
        while True:
            print("Тип профильной функции:")
            print("[1] Gaussian")
            print("[2] Lorentzian")
            print("[3] Voigt")
            mt_choice = ask_int("Ваш выбор (по умолчанию 1): ", default=1)
            models_type = {1: "GaussianModel", 2: "LorentzianModel", 3: "VoigtModel"}.get(
                mt_choice, "GaussianModel"
            )

            extra_components = ask_int("Дополнительные компоненты сверх найденных пиков (по умолчанию 4): ", default=4)
            amorph_components = ask_int("Сколько из них оставить под аморфный горб (по умолчанию 1): ", default=1)

            wmin = ask_int("Минимальная ширина пиков в точках (по умолчанию 3): ", default=3)
            wmax = ask_int("Максимальная ширина пиков в точках (по умолчанию 40): ", default=40)
            if wmax <= wmin:
                wmax = wmin + 1
            peak_widths_arr = (wmin, wmax)

            peaks_detected = generate_spec.detect_peaks(
                df_proc["two_theta"].values,
                df_proc["intensity"].values,
                width_range=peak_widths_arr,
                rel_height_min=CONFIG.rel_height_min,
                prominence_rel=0.02,
            )
            peak_indices = np.array([p.index for p in peaks_detected], dtype=int) if peaks_detected else np.array([], dtype=int)
            print(f"Найдено пиков: {len(peaks_detected)}")

            models_arr = generate_spec.build_model_from_peaks(
                peaks_detected,
                model_type=models_type,
                two_theta=df_proc["two_theta"].values,
                intensity=df_proc["intensity"].values,
                extra_components=extra_components,
                amorph_components=amorph_components,
            )
            spec = generate_spec.default_spec(df_proc, models_arr)
            print(
                f"Компоненты модели: всего={len(models_arr)} "
                f"(кристаллических={len(peaks_detected)}, аморфных={min(amorph_components, extra_components)}, "
                f"запасных={max(extra_components - min(amorph_components, extra_components), 0)})"
            )

            plot_with_peaks(spec["x"], spec["y"], peak_indices)

            next_action = input(
                "Что дальше? [1] Фит  [2] Изменить extra/amorph  [3] Изменить ширины поиска  [0] Назад: "
            ).strip()
            if next_action in ("1", ""):
                break
            if next_action == "0":
                # go back to preprocessing loop
                spec = None
                break
            if next_action == "2":
                continue
            if next_action == "3":
                continue

        if spec is None:
            continue

        # Fitting
        go_fit = input("Запустить аппроксимацию пиков? [Y/n]: ").strip().lower()
        if go_fit not in ("", "y", "yes"):
            print("Аппроксимация отменена.")
            continue

        model = fit_model.Model(spec)
        output = model.fit()
        peak_table = model.print_best_values()
        peak_table["d_hkl"] = peak_table["center"].apply(get_d_hkl)
        peak_table["crystal_size_nm"] = peak_table.apply(
            lambda row: get_crystallite_size_scherrer(row["fwhm"], row["center"]),
            axis=1,
        )
        if "is_amorphous" in peak_table.columns:
            peak_table["is_crystalline"] = ~peak_table["is_amorphous"].astype(bool)
        elif "kind" in peak_table.columns:
            peak_table["is_crystalline"] = peak_table["kind"].astype(str) == "crystalline"
        else:
            peak_table["is_crystalline"] = select_crystalline_peaks(
                peak_table,
                fwhm_min=CONFIG.ci_fwhm_min,
                fwhm_max=CONFIG.ci_fwhm_max,
                rel_height_min=CONFIG.rel_height_min,
            )

        ci = compute_ci(
            df_proc["two_theta"].values,
            df_proc["intensity"].values,
            peak_table,
            fwhm_min=CONFIG.ci_fwhm_min,
            fwhm_max=CONFIG.ci_fwhm_max,
            rel_height_min=CONFIG.rel_height_min,
        )
        print("Результаты подгонки:")
        print(peak_table)
        print(f"Индекс кристалличности CI = {ci:.1f} %")

        # Фазовый анализ
        do_phase = input("Выполнить фазовый анализ (CIF)? [y/N]: ").strip().lower()
        ref_db = []
        if do_phase in ("y", "yes"):
            ref_db = select_phases_menu(
                phases_dir="CIF",
                exp_two_theta=df_proc["two_theta"].values,
                exp_intensity=df_proc["intensity"].values,
                interval_2theta=(tmin, tmax),
            )
            if ref_db:
                ranking = search_match_all_phases(peak_table, ref_db=ref_db)
                print("\nРанжирование фаз по FoM:")
                print(ranking)

                peak_table = annotate_peaks_with_best_phase(
                    peak_table, ref_db=ref_db, delta_2theta_max=CONFIG.delta_2theta_max
                )
                print("\nПики с фазовой разметкой:")
                print(
                    peak_table[
                        ["prefix", "center", "fwhm", "crystal_size_nm", "phase_id", "dtheta_match", "intensity_diff"]
                    ]
                )
            else:
                print("Фазы не выбраны, фазовый анализ пропущен.")

        fitted_peaks = fitted_peaks_from_table(peak_table)
        assign_phases_to_peaks(fitted_peaks, ref_db=ref_db, max_delta_two_theta=CONFIG.delta_2theta_max)
        phase_results = aggregate_phase_results(fitted_peaks)
        fractions = compute_area_fractions(fitted_peaks)
        if fractions["area_total"] > 0:
            print("\nСводка по фазам (по площади):")
            header = f"{'Фаза':15s} {'N':>3s} {'Σплощадь':>10s} {'D_ср, нм':>10s} {'Доля,%':>9s} {'Крист.,%':>9s}"
            print(header)
            print("-" * len(header))
            for pr in sorted(phase_results, key=lambda p: p.area_total, reverse=True):
                area_pct = 100.0 * pr.area_total / fractions["area_total"] if fractions["area_total"] else 0.0
                cryst_pct = 0.0
                if pr.phase_id != "amorphous" and fractions["area_cryst"] > 0:
                    cryst_pct = 100.0 * pr.area_total / fractions["area_cryst"]
                d_avg = pr.mean_crystallite_size()
                d_str = f"{d_avg:.2f}" if d_avg is not None else "--"
                print(
                    f"{pr.phase_id:15s} {pr.n_peaks:3d} {pr.area_total:10.3f} {d_str:>10s} {area_pct:9.1f} {cryst_pct:9.1f}"
                )
            print("-" * len(header))
            print(
                f"Кристалличность: {fractions['x_cryst']*100:.1f} %, аморфность: {fractions['x_amorph']*100:.1f} %"
            )

        plot_fit_with_components(spec["x"], spec["y"], output, spec=spec)
        plot_fit_with_phase_markers(spec["x"], spec["y"], output.best_fit, peak_table)

        save_choice = input("Сохранить результаты в файлы? [Y/n]: ").strip().lower()
        if save_choice in ("", "y", "yes"):
            models_count = len(spec.get("model", []))
            fit_cfg = {
                "model_type": models_type,
                "models_count": models_count,
                "peak_widths": (wmin, wmax),
                "peak_method": "find_peaks",
                "rel_height_min": 0.05,
                "prominence_rel": 0.02,
            }
            save_results(
                spectr_path=spectr_path,
                spec=spec,
                output=output,
                peak_table=peak_table,
                ci=ci,
                preprocessing_cfg=preprocessing_cfg,
                fit_cfg=fit_cfg,
            )

        again = input("Обработать другой файл? [Y/n]: ").strip().lower()
        if again not in ("", "y", "yes"):
            break


if __name__ == "__main__":
    try:
        interactive_session()
    except KeyboardInterrupt:
        print("\nВыход по Ctrl+C")
        sys.exit(0)

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

from src.fitting import fit_model, generate_spec
from src.reading.data_from import spectrum_from_csv
from src.utils.peak_metrics import compute_ci
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
    plot_ref_preview,
)
from src.utils.peak_metrics import select_crystalline_peaks
from src.utils.search_match import (
    annotate_peaks_with_phases,
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

    # Fit plot
    fig, _ = plot_fit(spec["x"], spec["y"], output.best_fit, show=False)
    fig.savefig(result_dir / "fit.png", dpi=300)
    plt.close(fig)

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
            ref_df = build_ref_pattern_from_cif(str(cif_path), interval_2theta=(20.0, 60.0))
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
            tmin = ask_float("Минимальный 2θ (по умолчанию 20): ", default=tmin)
            tmax = ask_float("Максимальный 2θ (по умолчанию 60): ", default=tmax)
            if tmax <= tmin:
                print("Максимум должен быть больше минимума.")
                continue

            df_proc = trim_2theta(df_raw, interval=(tmin, tmax))

            # Savitzky–Golay options
            sg_choice = input(
                "Применить сглаживание Савицкого–Голея? [1]Да (31/3) [2]Да, вручную [0]Нет (Enter=1): "
            ).strip()
            if sg_choice in ("", "1"):
                wl, po = 31, 3
            elif sg_choice == "2":
                wl = ask_int("Окно сглаживания (нечетное, по умолчанию 31): ", default=31)
                po = ask_int("Порядок полинома (по умолчанию 3): ", default=3)
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

            models_count = ask_int("Максимальное число пиков (по умолчанию 40): ", default=40)

            wmin = ask_int("Минимальная ширина пиков в точках (по умолчанию 3): ", default=3)
            wmax = ask_int("Максимальная ширина пиков в точках (по умолчанию 40): ", default=40)
            if wmax <= wmin:
                wmax = wmin + 1
            peak_widths_arr = (wmin, wmax)

            models_arr = [{"type": models_type} for _ in range(models_count)]
            spec = generate_spec.default_spec(df_proc, models_arr)
            spec, peaks_found = generate_spec.update_spec_from_peaks(
                spec,
                list(range(models_count)),
                peak_widths=peak_widths_arr,
                method="find_peaks",
            )
            print(f"Найдено пиков: {len(peaks_found)}")

            plot_with_peaks(spec["x"], spec["y"], peaks_found)

            next_action = input(
                "Что дальше? [1] Фит  [2] Изменить число пиков  [3] Изменить ширины поиска  [0] Назад: "
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
        peak_table["is_crystalline"] = select_crystalline_peaks(peak_table)

        ci = compute_ci(df_proc["two_theta"].values, df_proc["intensity"].values, peak_table)
        print("Результаты подгонки:")
        print(peak_table)
        print(f"Индекс кристалличности CI = {ci:.1f} %")

        # Фазовый анализ
        do_phase = input("Выполнить фазовый анализ (CIF)? [y/N]: ").strip().lower()
        if do_phase in ("y", "yes"):
            ref_db = select_phases_menu(
                phases_dir="CIF",
                exp_two_theta=df_proc["two_theta"].values,
                exp_intensity=df_proc["intensity"].values,
            )
            if ref_db:
                ranking = search_match_all_phases(peak_table, ref_db=ref_db)
                print("\nРанжирование фаз по FoM:")
                print(ranking)

                matches = annotate_peaks_with_phases(peak_table, ref_db=ref_db)
                peak_table["phase_id"] = None
                for _, row in matches.iterrows():
                    exp_idx = row.get("exp_index")
                    if exp_idx in peak_table.index:
                        peak_table.at[exp_idx, "phase_id"] = row["phase_id"]
                print("\nПики с фазовой разметкой:")
                print(peak_table[["prefix", "center", "fwhm", "crystal_size_nm", "phase_id"]])
            else:
                print("Фазы не выбраны, фазовый анализ пропущен.")

        plot_fit_with_phase_markers(spec["x"], spec["y"], output.best_fit, peak_table)

        save_choice = input("Сохранить результаты в файлы? [Y/n]: ").strip().lower()
        if save_choice in ("", "y", "yes"):
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

import argparse
import sys
import uuid

import numpy as np

from src.config import CONFIG
from src.fitting import fit_model, generate_spec
from src.plotting.plots import plot_fit
from src.reading.data_from import spectrum_from_csv
from src.utils.peak_metrics import compute_ci, compute_total_area, select_crystalline_peaks
from src.utils.preprocessing import (
    apply_savgol,
    normalize_intensity,
    subtract_baseline,
    trim_2theta,
)
from src.utils.search_match import search_match_all_phases
from src.utils.xrd_geometry import get_crystallite_size_scherrer, get_d_hkl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XRD/Raman/FTIR spectrum processing and peak fitting tool."
    )
    parser.add_argument(
        "spectra_type",
        help="Choose spectrum type",
        type=str,
        choices=["xrd", "raman", "ftir"],
    )
    parser.add_argument(
        "-path",
        "--spectra_file_path",
        help="Relative or full path(s) to spectrum file(s) (csv format)",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-fit",
        "--fit_immediately",
        help="Fit without interactive confirmation",
        action="store_true",
    )
    parser.add_argument(
        "-m",
        "--model_type",
        help="Peak model type for lmfit",
        type=str,
        choices=["GaussianModel", "LorentzianModel", "VoigtModel"],
        default="GaussianModel",
    )
    parser.add_argument(
        "-n",
        "--models_count",
        help="Maximum number of peak models to allocate",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--peak_width_min",
        help="Minimum peak width (in points) for peak finder",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--peak_width_max",
        help="Maximum peak width (in points) for peak finder",
        type=int,
        default=30,
    )
    return parser.parse_args()


def process_spectrum(
    spectr_path: str,
    spectra_type: str,
    fit_immediately: bool,
    models_type: str,
    models_count: int,
    peak_widths: tuple[int, int],
) -> None:
    """Full pipeline for a single spectrum: read -> build spec -> find peaks -> (optional) fit."""
    spectr_name = spectr_path.split("/")[-1].split(".")[0]
    spectr_id = str(uuid.uuid4())

    print(f"\n=== Processing: {spectr_name} ===")
    print(f"Spectrum ID: {spectr_id}")
    print(f"Type: {spectra_type}")
    print(f"File: {spectr_path}")

    # TODO: при необходимости сделать разную предобработку для xrd/raman/ftir.
    if spectra_type != "xrd":
        print(f"[WARN] Non-XRD types ('{spectra_type}') пока обрабатываются как XRD по умолчанию.")

    try:
        df_raw = spectrum_from_csv(spectr_path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {spectr_path}", file=sys.stderr)
        return
    except Exception as exc:
        print(f"[ERROR] Failed to read spectrum '{spectr_path}': {exc}", file=sys.stderr)
        return

    if df_raw.empty:
        print(f"[WARN] Spectrum '{spectr_path}' is empty after loading.")
        return

    # Preprocessing: trim → smooth → baseline → normalize to 0–100 %
    preprocess_cfg = {
        "interval": (CONFIG.theta_min, CONFIG.theta_max),
        "savgol_window": 31,
        "savgol_poly": 3,
        "baseline": True,
        "normalize_0_100": True,
    }

    df = trim_2theta(df_raw, interval=preprocess_cfg["interval"])
    df["intensity"] = apply_savgol(
        df["intensity"], window_length=preprocess_cfg["savgol_window"], polyorder=preprocess_cfg["savgol_poly"]
    )
    if preprocess_cfg["baseline"]:
        df["intensity"] = subtract_baseline(df["two_theta"], df["intensity"])
    if preprocess_cfg["normalize_0_100"]:
        df["intensity"] = normalize_intensity(df["intensity"]) * 100

    print(
        "Preprocessing params:",
        preprocess_cfg,
    )
    print("First rows of preprocessed spectrum:")
    print(df.head())

    area_total = compute_total_area(df["two_theta"].values, df["intensity"].values)
    print(f"Total area (trapz): {area_total:.3f}")

    # Allocate models array
    models_arr = [{"type": models_type} for _ in range(models_count)]

    # Build spec dict from raw DataFrame
    spec = generate_spec.default_spec(df, models_arr)

    # Update spec with peak positions
    peak_widths_arr = (peak_widths[0], peak_widths[1])
    print(f"Peak search widths (points): {peak_widths_arr[0]}–{peak_widths_arr[1]}")
    spec, peaks_found = generate_spec.update_spec_from_peaks(
        spec,
        list(range(models_count)),
        peak_widths=peak_widths_arr,
        method="find_peaks",
    )

    print(f"Found peaks: {len(peaks_found)}")

    # Fitting block
    if fit_immediately:
        do_fit = True
    else:
        go_fit = input("Start fitting? [y/n]: ").strip().lower()
        do_fit = go_fit in ("y", "yes")

    if not do_fit:
        print("Skip fitting for this spectrum.")
        return

    print("Start fitting...")
    model = fit_model.Model(spec)
    output = model.fit()
    peak_table = model.print_best_values()
    peak_table["d_hkl"] = peak_table["center"].apply(get_d_hkl)
    peak_table["crystal_size_nm"] = peak_table.apply(
        lambda row: get_crystallite_size_scherrer(row["fwhm"], row["center"]), axis=1
    )
    peak_table["is_crystalline"] = select_crystalline_peaks(peak_table)

    ci_value = compute_ci(
        df["two_theta"].values,
        df["intensity"].values,
        peak_table,
        fwhm_min=CONFIG.ci_fwhm_min,
        fwhm_max=CONFIG.ci_fwhm_max,
        rel_height_min=CONFIG.rel_height_min,
    )

    print("Best-fit parameters summary:")
    print(peak_table)
    print(f"Crystallinity Index (CI, %): {ci_value:.2f}")

    # Search–match (placeholder ref_db list for now)
    ref_db: list = []
    search_results = search_match_all_phases(peak_table, ref_db=ref_db)
    if search_results.empty:
        print("Search–match: no reference phases provided.")
    else:
        print("Search–match ranking:")
        print(search_results)

    # Plot result
    plot_fit(spec["x"], spec["y"], output.best_fit)


def main() -> None:
    args = parse_args()

    print("Hello! XRD Explorer is working.")
    print(f"spectra_type     : {args.spectra_type}")
    print(f"spectra_file_path: {args.spectra_file_path}")
    print(f"fit_immediately  : {args.fit_immediately}")
    print(f"model_type       : {args.model_type}")
    print(f"models_count     : {args.models_count}")

    peak_widths = (args.peak_width_min, args.peak_width_max)

    for spectr_path in args.spectra_file_path:
        process_spectrum(
            spectr_path=spectr_path,
            spectra_type=args.spectra_type,
            fit_immediately=args.fit_immediately,
            models_type=args.model_type,
            models_count=args.models_count,
            peak_widths=peak_widths,
        )


if __name__ == "__main__":
    main()

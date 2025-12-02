import argparse
import sys
import uuid

import numpy as np

from src.analysis.phase_assign import assign_phases_to_peaks
from src.analysis.phase_summary import aggregate_phase_results, compute_area_fractions
from src.config import CONFIG
from src.fitting import fit_model, generate_spec
from src.plotting.plots import plot_fit, plot_fit_with_components, plot_residuals
from src.reading.data_from import spectrum_from_csv
from src.utils.peak_metrics import (
    compute_ci,
    compute_total_area,
    fitted_peaks_from_table,
    select_crystalline_peaks,
    reclassify_amorphous_by_width,
)
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
        help="Deprecated: legacy cap on number of models (auto-peaks mode ignores unless provided).",
        type=int,
        default=None,
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
    parser.add_argument(
        "--extra-components",
        help="Number of extra components to add on top of detected peaks.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--amorph-components",
        help="How many of the extra components are reserved for amorphous broad peaks.",
        type=int,
        default=1,
    )
    return parser.parse_args()


def process_spectrum(
    spectr_path: str,
    spectra_type: str,
    fit_immediately: bool,
    models_type: str,
    models_count: int | None,
    peak_widths: tuple[int, int],
    extra_components: int,
    amorph_components: int,
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

    peak_widths_arr = (peak_widths[0], peak_widths[1])
    print(f"Peak search widths (points): {peak_widths_arr[0]}–{peak_widths_arr[1]}")
    peaks_detected = generate_spec.detect_peaks(
        df["two_theta"].values,
        df["intensity"].values,
        width_range=peak_widths_arr,
        rel_height_min=CONFIG.rel_height_min,
        prominence_rel=0.02,
    )
    print(f"Found peaks: {len(peaks_detected)}")

    if models_count is not None:
        allowed_extra = max(models_count - len(peaks_detected), 0)
        if allowed_extra < extra_components:
            print(f"[WARN] Extra components capped by legacy --models_count={models_count}.")
        extra_effective = min(extra_components, allowed_extra)
    else:
        extra_effective = extra_components

    models_arr = generate_spec.build_model_from_peaks(
        peaks_detected,
        model_type=models_type,
        two_theta=df["two_theta"].values,
        intensity=df["intensity"].values,
        extra_components=extra_effective,
        amorph_components=amorph_components,
    )
    total_components = len(models_arr)
    amorph_used = min(amorph_components, extra_effective)
    print(
        f"Model components: total={total_components} "
        f"(crystalline={len(peaks_detected)}, amorphous={amorph_used}, extra_narrow={extra_effective - amorph_used})"
    )

    # Build spec dict from raw DataFrame
    spec = generate_spec.default_spec(df, models_arr)

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
    residual_var = float(np.var(output.residual))
    print(f"Residual variance: {residual_var:.4f}")
    peak_table = model.print_best_values()
    peak_table["d_hkl"] = peak_table["center"].apply(get_d_hkl)
    peak_table["crystal_size_nm"] = peak_table.apply(
        lambda row: get_crystallite_size_scherrer(row["fwhm"], row["center"]), axis=1
    )

    peak_table, fwhm_thr = reclassify_amorphous_by_width(peak_table)
    print(f"FWHM threshold for amorphous reclass: {fwhm_thr:.2f} deg 2θ")

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

    fitted_peaks = fitted_peaks_from_table(peak_table)
    assign_phases_to_peaks(
        fitted_peaks,
        ref_db=ref_db,
        max_delta_two_theta=CONFIG.delta_2theta_max,
    )
    phase_results = aggregate_phase_results(fitted_peaks)
    fractions = compute_area_fractions(fitted_peaks)

    if fractions["area_total"] > 0:
        print("\nPhase summary (area-based):")
        header = f"{'Phase':15s} {'N':>3s} {'Area':>10s} {'D_avg (nm)':>12s} {'Share %':>9s} {'Cryst %':>9s}"
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
                f"{pr.phase_id:15s} {pr.n_peaks:3d} {pr.area_total:10.3f} {d_str:>12s} "
                f"{area_pct:9.1f} {cryst_pct:9.1f}"
            )
        print("-" * len(header))
        print(
            f"Crystalline fraction: {fractions['x_cryst']*100:.1f} %, "
            f"Amorphous: {fractions['x_amorph']*100:.1f} %"
        )

    # Plot result with components
    plot_fit_with_components(spec["x"], spec["y"], output, spec=spec, peak_table=peak_table)
    plot_residuals(
        spec["x"],
        output.residual,
        title=f"Остатки (var={residual_var:.4f})",
    )


def main() -> None:
    args = parse_args()

    print("Hello! XRD Explorer is working.")
    print(f"spectra_type     : {args.spectra_type}")
    print(f"spectra_file_path: {args.spectra_file_path}")
    print(f"fit_immediately  : {args.fit_immediately}")
    print(f"model_type       : {args.model_type}")
    print(f"models_count     : {args.models_count} (deprecated cap)")
    print(f"extra_components : {args.extra_components}")
    print(f"amorph_components: {args.amorph_components}")

    peak_widths = (args.peak_width_min, args.peak_width_max)

    for spectr_path in args.spectra_file_path:
        process_spectrum(
            spectr_path=spectr_path,
            spectra_type=args.spectra_type,
            fit_immediately=args.fit_immediately,
            models_type=args.model_type,
            models_count=args.models_count,
            peak_widths=peak_widths,
            extra_components=args.extra_components,
            amorph_components=args.amorph_components,
        )


if __name__ == "__main__":
    main()

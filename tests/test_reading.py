import pandas as pd

from src.reading.data_from import spectrum_from_csv


def test_spectrum_from_csv_no_header(tmp_path):
    path = tmp_path / "spec.csv"
    path.write_text("\n".join(["20,100", "21,110", "22,120"]))

    df = spectrum_from_csv(str(path))

    assert list(df.columns) == ["two_theta", "intensity"]
    assert len(df) == 3
    assert df["two_theta"].iloc[0] == 20
    assert df["intensity"].iloc[-1] == 120


def test_spectrum_from_csv_with_header_and_options(tmp_path):
    path = tmp_path / "spec_semicolon.csv"
    lines = ["t;I", "30;5", "31;6", "40;1"]
    path.write_text("\n".join(lines))

    df = spectrum_from_csv(
        str(path),
        sep=";",
        has_header=True,
        theta_col="t",
        int_col="I",
        interval_2theta=(30.0, 35.0),
        filter=True,
        window_length=3,
        polyorder=1,
        normalize=True,
        normalize_scale=10.0,
    )

    assert list(df.columns) == ["two_theta", "intensity"]
    assert df["two_theta"].between(30, 35).all()
    assert df["intensity"].max() <= 10.0 + 1e-6
    assert len(df) == 2  # cropped to 30–35

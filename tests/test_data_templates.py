"""Tests for raw data validation and empty template generation."""
import pandas as pd

from src.data import templates


def test_validate_raw_data_files_creates_empty_templates(tmp_path, capsys):
    raw_dir = tmp_path / "raw"

    all_present = templates.validate_raw_data_files(raw_dir=raw_dir)

    assert all_present is False
    output = capsys.readouterr().out
    assert "Missing raw CSV files detected" in output
    assert "No paid APIs are required" in output

    for spec in templates.RAW_DATA_SPECS:
        path = raw_dir / spec.filename
        assert path.exists()
        frame = pd.read_csv(path)
        assert list(frame.columns) == list(spec.columns)
        assert frame.empty


def test_validate_raw_data_files_does_not_overwrite_existing_files(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    existing = raw_dir / "elo_ratings.csv"
    existing.write_text("date,team,elo_rating\n2026-05-01,Australia,1500\n")

    templates.validate_raw_data_files(raw_dir=raw_dir, verbose=False)

    frame = pd.read_csv(existing)
    assert len(frame) == 1
    assert frame.loc[0, "team"] == "Australia"


def test_missing_specs_reports_only_absent_files(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    first = templates.RAW_DATA_SPECS[0]
    templates.write_template(first, raw_dir=raw_dir)

    missing = templates.missing_specs(raw_dir=raw_dir)

    assert first not in missing
    assert len(missing) == len(templates.RAW_DATA_SPECS) - 1


def test_required_validation_does_not_create_optional_templates(tmp_path):
    raw_dir = tmp_path / "raw"

    templates.validate_raw_data_files(raw_dir=raw_dir, verbose=False)

    for spec in templates.OPTIONAL_RAW_DATA_SPECS:
        assert not (raw_dir / spec.filename).exists()

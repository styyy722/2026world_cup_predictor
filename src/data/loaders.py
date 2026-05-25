"""Load raw CSV inputs from ``data/raw/``.

Every loader is defensive: if a required file is absent we create an empty
schema template and print source-specific download/collection instructions.
The first version is intentionally CSV-first and does not require paid APIs.
"""
from __future__ import annotations

import pandas as pd

from .. import config
from . import templates


def validate_raw_data_files() -> bool:
    """Check required raw CSV files and create empty templates if needed."""
    return templates.validate_raw_data_files()


def _ensure_inputs_exist() -> None:
    """Create empty template CSVs for any missing raw input files."""
    if templates.missing_specs():
        templates.validate_raw_data_files()


def load_results() -> pd.DataFrame:
    """Load international match results history."""
    _ensure_inputs_exist()
    df = pd.read_csv(config.RESULTS_FILE, parse_dates=["date"])
    df["neutral"] = df["neutral"].astype(bool)
    return df.sort_values("date").reset_index(drop=True)


def load_fifa_rankings() -> pd.DataFrame:
    """Load FIFA rankings (latest row per team is used for fixtures)."""
    _ensure_inputs_exist()
    return pd.read_csv(config.FIFA_RANKINGS_FILE, parse_dates=["date"])


def load_elo() -> pd.DataFrame:
    """Load Elo ratings."""
    _ensure_inputs_exist()
    return pd.read_csv(config.ELO_FILE, parse_dates=["date"])


def load_fixtures() -> pd.DataFrame:
    """Load the 2026 group-stage fixtures."""
    _ensure_inputs_exist()
    df = pd.read_csv(config.FIXTURES_FILE, parse_dates=["date"])
    df["neutral"] = df["neutral"].astype(bool)
    return df


def load_groups() -> pd.DataFrame:
    """Load the 2026 group assignments."""
    _ensure_inputs_exist()
    return pd.read_csv(config.GROUPS_FILE)


def load_team_context() -> pd.DataFrame:
    """Load optional team context features, or an empty schema if absent."""
    spec = templates.optional_spec("team_context.csv")
    path = templates.expected_path(spec)
    if not path.exists():
        return templates.make_template(spec)
    return pd.read_csv(path, parse_dates=["date"])


def load_player_status() -> pd.DataFrame:
    """Load optional player availability/status snapshots."""
    spec = templates.optional_spec("player_status.csv")
    path = templates.expected_path(spec)
    if not path.exists():
        return templates.make_template(spec)
    return pd.read_csv(path, parse_dates=["as_of_date", "expected_return"])


def load_player_form() -> pd.DataFrame:
    """Load optional player form/performance snapshots."""
    spec = templates.optional_spec("player_form.csv")
    path = templates.expected_path(spec)
    if not path.exists():
        return templates.make_template(spec)
    return pd.read_csv(path, parse_dates=["date"])


def load_team_status() -> pd.DataFrame:
    """Load optional team-level experience and stability snapshots."""
    spec = templates.optional_spec("team_status.csv")
    path = templates.expected_path(spec)
    if not path.exists():
        return templates.make_template(spec)
    return pd.read_csv(path, parse_dates=["as_of_date"])


def load_match_context() -> pd.DataFrame:
    """Load optional fixture weather, venue, and travel context."""
    spec = templates.optional_spec("match_context.csv")
    path = templates.expected_path(spec)
    if not path.exists():
        return templates.make_template(spec)
    return pd.read_csv(path, parse_dates=["date"])


def load_betting_odds() -> pd.DataFrame:
    """Load optional manually collected betting odds, or an empty schema."""
    spec = templates.optional_spec("betting_odds.csv")
    path = templates.expected_path(spec)
    if not path.exists():
        return templates.make_template(spec)
    return pd.read_csv(path, parse_dates=["date"])


def latest_ratings() -> pd.DataFrame:
    """Return one row per team with the most recent Elo + FIFA ranking.

    Used to attach ratings to the 2026 fixtures, which have no history.
    """
    elo = load_elo().sort_values("date").groupby("team", as_index=False).last()
    fifa = (
        load_fifa_rankings()
        .sort_values("date")
        .groupby("team", as_index=False)
        .last()[["team", "fifa_rank", "fifa_points"]]
    )
    merged = elo.merge(fifa, on="team", how="outer")
    # Fill any gaps with neutral defaults so simulation never crashes.
    merged["elo_rating"] = merged["elo_rating"].fillna(config.DEFAULT_ELO)
    merged["fifa_rank"] = merged["fifa_rank"].fillna(merged["fifa_rank"].max())
    merged["fifa_points"] = merged["fifa_points"].fillna(merged["fifa_points"].min())
    return merged[["team", "elo_rating", "fifa_rank", "fifa_points"]]

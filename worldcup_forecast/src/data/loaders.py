"""Load raw CSV inputs, generating templates if files are missing.

Every loader is defensive: if a required file is absent we create a template
(via templates.write_all_templates) and print a helpful message rather than
crashing. This keeps the MVP runnable out of the box.
"""
from __future__ import annotations

import pandas as pd

from .. import config
from . import templates


def _ensure_inputs_exist() -> None:
    """Create template CSVs for any missing raw input files."""
    required = [
        config.RESULTS_FILE,
        config.FIFA_RANKINGS_FILE,
        config.ELO_FILE,
        config.FIXTURES_FILE,
        config.GROUPS_FILE,
    ]
    if any(not p.exists() for p in required):
        print("[loaders] One or more raw data files missing - generating templates.")
        templates.write_all_templates()


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

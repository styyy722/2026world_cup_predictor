"""Raw CSV schema templates for the data pipeline.

The first version of the project expects users to place CSV files in
``data/raw/``. When a required file is absent, we create an empty CSV with the
right columns and print source-specific collection instructions. This keeps the
pipeline free of paid API dependencies while making the required manual inputs
explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .. import config


@dataclass(frozen=True)
class RawDataSpec:
    """Expected file metadata for one raw input CSV."""

    filename: str
    columns: tuple[str, ...]
    source: str
    instructions: tuple[str, ...]


RAW_DATA_SPECS: tuple[RawDataSpec, ...] = (
    RawDataSpec(
        filename="international_results.csv",
        columns=(
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "neutral",
            "country",
        ),
        source=(
            "Kaggle: International Football Results from 1872 to present "
            "(https://www.kaggle.com/datasets/martj42/"
            "international-football-results-from-1872-to-2017)"
        ),
        instructions=(
            "Download the Kaggle dataset and use the results CSV.",
            "Keep one row per international match and rename the file to "
            "international_results.csv.",
            "Use ISO dates (YYYY-MM-DD) and boolean neutral values.",
        ),
    ),
    RawDataSpec(
        filename="fifa_rankings.csv",
        columns=("date", "team", "fifa_rank", "fifa_points"),
        source=(
            "FIFA official men's ranking page for the current snapshot, plus "
            "Kaggle or GitHub historical FIFA ranking snapshots for older dates"
        ),
        instructions=(
            "Collect the latest ranking from FIFA's official men's ranking page: "
            "https://inside.fifa.com/fifa-world-ranking/men?lv=true.",
            "Append older ranking snapshots from a public Kaggle or GitHub "
            "historical FIFA rankings dataset such as "
            "https://www.kaggle.com/datasets/lucasyukioimafuko/"
            "fifa-mens-world-ranking/versions/2 or "
            "https://github.com/Dato-Futbol/fifa-ranking.",
            "Standardise columns to date, team, fifa_rank, fifa_points.",
        ),
    ),
    RawDataSpec(
        filename="elo_ratings.csv",
        columns=("date", "team", "elo_rating"),
        source=(
            "World Football Elo Ratings and/or a public Kaggle historical "
            "international football Elo dataset"
        ),
        instructions=(
            "Download or manually collect international team Elo ratings from "
            "https://www.eloratings.net or a public Kaggle dataset such as "
            "https://www.kaggle.com/datasets/saifalnimri/"
            "international-football-elo-ratings.",
            "If the source column is named rating or elo, rename it to "
            "elo_rating.",
            "Keep one row per team per rating snapshot date.",
        ),
    ),
    RawDataSpec(
        filename="world_cup_2026_fixtures.csv",
        columns=(
            "match_id",
            "stage",
            "group",
            "date",
            "team_a",
            "team_b",
            "venue",
            "neutral",
        ),
        source=(
            "FIFA official World Cup 2026 match schedule page "
            "(https://www.fifa.com/en/tournaments/mens/worldcup/"
            "canadamexicousa2026/articles/"
            "match-schedule-fixtures-results-teams-stadiums)"
        ),
        instructions=(
            "Manually collect the 2026 group-stage fixtures from FIFA's "
            "official schedule.",
            "Use stage='group' for group-stage matches.",
            "Set neutral=True unless you intentionally model host advantage.",
        ),
    ),
    RawDataSpec(
        filename="world_cup_2026_groups.csv",
        columns=("group", "team"),
        source=(
            "FIFA official World Cup 2026 final draw results page "
            "(https://www.fifa.com/en/tournaments/mens/worldcup/"
            "canadamexicousa2026/articles/final-draw-results)"
        ),
        instructions=(
            "Manually collect the final draw group assignments from FIFA.",
            "Use group letters A through L and one row per team.",
            "Keep team names consistent with the other raw CSV files.",
        ),
    ),
)

OPTIONAL_RAW_DATA_SPECS: tuple[RawDataSpec, ...] = (
    RawDataSpec(
        filename="team_context.csv",
        columns=(
            "date",
            "team",
            "injured_players",
            "suspended_players",
            "squad_market_value_eur",
            "xg_for_10",
            "xg_against_10",
        ),
        source=(
            "Optional manually audited team context: injury/suspension notes, "
            "squad market value, and recent expected-goals summaries"
        ),
        instructions=(
            "Keep one row per team per snapshot date.",
            "Use public sources only unless you have a licensed private source.",
            "Leave the file absent if you do not want to use these features yet.",
        ),
    ),
    RawDataSpec(
        filename="player_status.csv",
        columns=(
            "as_of_date",
            "team",
            "player",
            "position",
            "club",
            "squad_status",
            "availability_status",
            "injury_type",
            "expected_return",
            "is_probable_starter",
            "source_url",
        ),
        source=(
            "Optional player availability from FIFA squad announcements, "
            "national federation squad pages, Transfermarkt injury history, "
            "RotoWire, Sports Mole, or SportsGambler"
        ),
        instructions=(
            "Keep one row per player per status snapshot date.",
            "Use squad_status for final/provisional/replaced/out-of-squad states.",
            "Use availability_status values such as available, doubtful, "
            "injured, suspended, or out.",
            "Keep source_url populated so late injury/news updates are auditable.",
        ),
    ),
    RawDataSpec(
        filename="player_form.csv",
        columns=(
            "date",
            "season",
            "team",
            "player",
            "club",
            "competition",
            "minutes",
            "starts",
            "goals",
            "assists",
            "xg",
            "xa",
            "cards",
            "source_url",
        ),
        source=(
            "Optional club/international player form from FBref, Statbunker, "
            "Kaggle FBref-derived files, or manually audited public sources"
        ),
        instructions=(
            "Keep one row per player per season or snapshot date.",
            "Use minutes and xG/xA when available; leave unknown values blank.",
            "Prefer public CSV exports or manually reviewed tables over paid APIs.",
        ),
    ),
    RawDataSpec(
        filename="team_status.csv",
        columns=(
            "as_of_date",
            "team",
            "average_age",
            "total_caps",
            "coach_tenure_days",
            "fifa_confederation",
            "source_url",
        ),
        source=(
            "Optional team status from FIFA team pages, national federation "
            "profiles, Transfermarkt squad pages, or Statbunker"
        ),
        instructions=(
            "Keep one row per team per status snapshot date.",
            "Use average_age, total_caps, and coach_tenure_days as squad "
            "experience/stability proxies.",
            "Leave fifa_confederation as text; the current model uses the "
            "numeric stability fields.",
        ),
    ),
    RawDataSpec(
        filename="match_context.csv",
        columns=(
            "match_id",
            "date",
            "team_a",
            "team_b",
            "venue",
            "city",
            "temperature_c",
            "humidity_pct",
            "wind_kmh",
            "altitude_m",
            "team_a_travel_km",
            "team_b_travel_km",
            "source_url",
        ),
        source=(
            "Optional match context from FIFA schedule/venue pages plus free "
            "weather and geography sources such as Open-Meteo"
        ),
        instructions=(
            "Keep one row per fixture when venue/weather/travel estimates are known.",
            "Use neutral defaults if weather forecasts are not yet available.",
            "Do not require a paid API for the first version.",
        ),
    ),
    RawDataSpec(
        filename="betting_odds.csv",
        columns=(
            "match_id",
            "date",
            "team_a",
            "team_b",
            "team_a_decimal_odds",
            "draw_decimal_odds",
            "team_b_decimal_odds",
            "bookmaker",
        ),
        source=(
            "Optional manually collected bookmaker decimal odds. Paid odds APIs "
            "are not required."
        ),
        instructions=(
            "Use decimal odds and keep team_a/team_b aligned to the fixture file.",
            "The predictor normalises implied probabilities before blending.",
            "Leave the file absent unless you choose to blend odds later.",
        ),
    ),
)


def expected_path(spec: RawDataSpec, raw_dir: Path | None = None) -> Path:
    """Return the filesystem path for a raw data spec."""
    return (raw_dir or config.RAW_DIR) / spec.filename


def make_template(spec: RawDataSpec) -> pd.DataFrame:
    """Return an empty template DataFrame with the expected columns."""
    return pd.DataFrame(columns=list(spec.columns))


def write_template(spec: RawDataSpec, raw_dir: Path | None = None) -> Path:
    """Write one empty template CSV and return its path."""
    path = expected_path(spec, raw_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_template(spec).to_csv(path, index=False)
    return path


def missing_specs(raw_dir: Path | None = None) -> list[RawDataSpec]:
    """Return specs whose expected CSV file does not exist."""
    return [spec for spec in RAW_DATA_SPECS if not expected_path(spec, raw_dir).exists()]


def _display_path(path: Path) -> str:
    """Return a readable path, relative to the project when possible."""
    try:
        return str(path.relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path)


def print_missing_data_instructions(
    missing: list[RawDataSpec],
    raw_dir: Path | None = None,
    created_paths: list[Path] | None = None,
) -> None:
    """Print clear manual download/collection instructions for missing files."""
    if not missing:
        print("[data] All required raw CSV files are present.")
        return

    print("[data] Missing raw CSV files detected.")
    if created_paths:
        print("[data] Empty template CSVs were created for the missing files.")
    print("[data] Populate these templates before running training/simulation:")

    created = {p.name for p in created_paths or []}
    for spec in missing:
        path = expected_path(spec, raw_dir)
        status = "created" if path.name in created else "missing"
        print(f"\n- {_display_path(path)} ({status})")
        print(f"  Expected columns: {', '.join(spec.columns)}")
        print(f"  Source: {spec.source}")
        for step in spec.instructions:
            print(f"  - {step}")

    print(
        "\n[data] No paid APIs are required for this version. Optional API "
        "connectors can be added later after the CSV workflow is stable."
    )


def validate_raw_data_files(
    raw_dir: Path | None = None,
    create_missing: bool = True,
    verbose: bool = True,
) -> bool:
    """Validate required raw CSV existence and optionally create templates.

    Returns ``True`` when every required file already existed, and ``False`` if
    any file was missing at validation time.
    """
    missing = missing_specs(raw_dir)
    created_paths: list[Path] = []

    if missing and create_missing:
        for spec in missing:
            created_paths.append(write_template(spec, raw_dir))

    if verbose:
        print_missing_data_instructions(missing, raw_dir, created_paths)

    return not missing


def write_all_templates(raw_dir: Path | None = None) -> None:
    """Create empty template CSVs for any required raw files that are missing."""
    validate_raw_data_files(raw_dir=raw_dir, create_missing=True, verbose=True)


def make_groups_template() -> pd.DataFrame:
    """Return the empty World Cup groups template."""
    spec = next(s for s in RAW_DATA_SPECS if s.filename == "world_cup_2026_groups.csv")
    return make_template(spec)


def make_elo_template() -> pd.DataFrame:
    """Return the empty Elo ratings template."""
    spec = next(s for s in RAW_DATA_SPECS if s.filename == "elo_ratings.csv")
    return make_template(spec)


def make_fifa_rankings_template() -> pd.DataFrame:
    """Return the empty FIFA rankings template."""
    spec = next(s for s in RAW_DATA_SPECS if s.filename == "fifa_rankings.csv")
    return make_template(spec)


def make_results_template() -> pd.DataFrame:
    """Return the empty international results template."""
    spec = next(s for s in RAW_DATA_SPECS if s.filename == "international_results.csv")
    return make_template(spec)


def make_fixtures_template() -> pd.DataFrame:
    """Return the empty 2026 fixtures template."""
    spec = next(s for s in RAW_DATA_SPECS if s.filename == "world_cup_2026_fixtures.csv")
    return make_template(spec)


def optional_spec(filename: str) -> RawDataSpec:
    """Return metadata for an optional raw CSV."""
    return next(s for s in OPTIONAL_RAW_DATA_SPECS if s.filename == filename)

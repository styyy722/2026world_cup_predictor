"""Central configuration: paths and tournament constants.

Keeping paths and format constants in one place avoids hardcoding them
across modules and makes the 2026 tournament structure explicit.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# config.py lives in <project>/src/, so the project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
CHARTS_DIR = OUTPUTS_DIR / "charts"

MODELS_DIR = PROJECT_ROOT / "models_store"

# Expected raw input files (see README for schemas).
RESULTS_FILE = RAW_DIR / "international_results.csv"
FIFA_RANKINGS_FILE = RAW_DIR / "fifa_rankings.csv"
ELO_FILE = RAW_DIR / "elo_ratings.csv"
FIXTURES_FILE = RAW_DIR / "world_cup_2026_fixtures.csv"
GROUPS_FILE = RAW_DIR / "world_cup_2026_groups.csv"

# ---------------------------------------------------------------------------
# 2026 FIFA World Cup format
# ---------------------------------------------------------------------------
# The 2026 World Cup expands to 48 teams arranged in 12 groups of 4.
# The top 2 teams of every group (24 teams) advance automatically, plus the
# 8 best third-placed teams, giving a 32-team Round of 32 knockout bracket.
N_TEAMS = 48
N_GROUPS = 12
TEAMS_PER_GROUP = 4
N_THIRD_PLACE_QUALIFIERS = 8  # best third-placed teams that reach the R32

# Group-stage points.
POINTS_WIN = 3
POINTS_DRAW = 1
POINTS_LOSS = 0

# Knockout stage labels in order of progression.
KNOCKOUT_STAGES = ["R32", "R16", "QF", "SF", "F"]

# Default Elo for a team with no rating available.
DEFAULT_ELO = 1500.0

# Training scope. The default model intentionally avoids stale international
# history because squads and player availability change substantially cycle to
# cycle.
TRAINING_LOOKBACK_YEARS = 4
LAST_WORLD_CUP_YEAR = 2022

# Default simulation counts.
QUICK_SIMULATIONS = 10_000
FINAL_SIMULATIONS = 100_000


def ensure_dirs() -> None:
    """Create all output/model directories if they do not yet exist."""
    for d in (RAW_DIR, PROCESSED_DIR, TABLES_DIR, CHARTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

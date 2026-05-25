"""Generate example/template CSV files when real data is missing.

If a user has not yet supplied real data, we still want the pipeline to run
end-to-end so they can see the expected schema and a plausible result. These
templates contain a small but self-consistent synthetic dataset:

* 48 plausible national teams arranged into 12 groups,
* synthetic Elo ratings and FIFA rankings,
* a synthetic match history,
* a full 2026 group-stage fixture list (72 matches).

Assumptions are made explicit in comments. Real data should replace these.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config

# ---------------------------------------------------------------------------
# A plausible 48-team field (illustrative only, not an official draw).
# Each entry: team name and a "true strength" used to generate synthetic data.
# ---------------------------------------------------------------------------
_TEAM_STRENGTHS: dict[str, float] = {
    # Group A
    "Mexico": 1680, "Canada": 1560, "Morocco": 1700, "Japan": 1640,
    # Group B
    "USA": 1640, "Wales": 1560, "Senegal": 1620, "South Korea": 1590,
    # Group C
    "Argentina": 1990, "Australia": 1530, "Poland": 1560, "Ecuador": 1580,
    # Group D
    "France": 1950, "Denmark": 1660, "Tunisia": 1500, "Peru": 1520,
    # Group E
    "Spain": 1900, "Costa Rica": 1490, "Germany": 1830, "Serbia": 1600,
    # Group F
    "Belgium": 1780, "Croatia": 1750, "Saudi Arabia": 1480, "Nigeria": 1580,
    # Group G
    "Brazil": 1960, "Switzerland": 1690, "Cameroon": 1540, "Sweden": 1620,
    # Group H
    "Portugal": 1870, "Ghana": 1520, "Uruguay": 1720, "Iran": 1540,
    # Group I
    "Netherlands": 1840, "Qatar": 1450, "Egypt": 1560, "Chile": 1570,
    # Group J
    "England": 1900, "Ukraine": 1620, "Algeria": 1550, "Colombia": 1680,
    # Group K
    "Italy": 1830, "Norway": 1640, "Ivory Coast": 1560, "Panama": 1470,
    # Group L
    "Uruguay B": 1500, "Turkey": 1600, "Austria": 1630, "Scotland": 1560,
}

_GROUP_NAMES = list("ABCDEFGHIJKL")  # 12 groups


def _team_groups() -> pd.DataFrame:
    """Assign the 48 teams to 12 groups of 4, in listed order."""
    teams = list(_TEAM_STRENGTHS.keys())
    rows = []
    for i, team in enumerate(teams):
        group = _GROUP_NAMES[i // config.TEAMS_PER_GROUP]
        rows.append({"group": group, "team": team})
    return pd.DataFrame(rows)


def make_groups_template() -> pd.DataFrame:
    return _team_groups()


def make_elo_template(as_of: str = "2026-05-01") -> pd.DataFrame:
    """Synthetic Elo ratings ~ true strength with small noise."""
    rng = np.random.default_rng(42)
    rows = []
    for team, strength in _TEAM_STRENGTHS.items():
        elo = float(strength + rng.normal(0, 15))
        rows.append({"date": as_of, "team": team, "elo_rating": round(elo, 1)})
    return pd.DataFrame(rows)


def make_fifa_rankings_template(as_of: str = "2026-05-01") -> pd.DataFrame:
    """Synthetic FIFA rankings derived from strength ordering."""
    ordered = sorted(_TEAM_STRENGTHS.items(), key=lambda kv: -kv[1])
    rows = []
    for rank, (team, strength) in enumerate(ordered, start=1):
        # FIFA points roughly scale with strength.
        points = round(float(strength) - 600 + 400, 1)
        rows.append(
            {"date": as_of, "team": team, "fifa_rank": rank, "fifa_points": points}
        )
    return pd.DataFrame(rows)


def make_results_template(n_matches: int = 2200) -> pd.DataFrame:
    """Synthetic international match history.

    Goals are drawn from Poisson distributions whose means depend on the Elo
    gap between the teams, so that stronger teams win more often. This gives
    the logistic model and backtester something realistic to learn from.
    """
    rng = np.random.default_rng(7)
    teams = list(_TEAM_STRENGTHS.keys())
    strengths = _TEAM_STRENGTHS

    tournaments = ["Friendly", "FIFA World Cup", "UEFA Euro",
                   "Copa America", "World Cup qualification", "Nations League"]
    # Weight friendlies more heavily, as in real history.
    t_weights = np.array([0.45, 0.08, 0.07, 0.07, 0.23, 0.10])

    # Spread matches over ~23 years so backtests have training data before
    # every World Cup from 2006 onward (2006/2010/2014/2018/2022).
    dates = pd.date_range("2003-01-01", "2026-04-01", periods=n_matches)

    rows = []
    for i in range(n_matches):
        home, away = rng.choice(teams, size=2, replace=False)
        elo_diff = (strengths[home] - strengths[away])
        neutral = bool(rng.random() < 0.3)
        # Home advantage adds to the effective Elo gap unless neutral.
        eff_diff = elo_diff + (0 if neutral else 60)
        # Expected goals from a logistic-ish mapping of the gap.
        base = 1.35
        lam_home = base * np.exp(eff_diff / 600)
        lam_away = base * np.exp(-eff_diff / 600)
        hs = int(rng.poisson(max(0.15, lam_home)))
        as_ = int(rng.poisson(max(0.15, lam_away)))
        tournament = str(rng.choice(tournaments, p=t_weights))
        rows.append({
            "date": dates[i].strftime("%Y-%m-%d"),
            "home_team": home,
            "away_team": away,
            "home_score": hs,
            "away_score": as_,
            "tournament": tournament,
            "neutral": neutral,
            "country": "Neutral" if neutral else home,
        })
    return pd.DataFrame(rows)


def make_fixtures_template() -> pd.DataFrame:
    """Full 2026 group-stage fixture list: 12 groups x 6 matches = 72 games.

    Each group of 4 plays a single round-robin (6 matches per group). Only the
    group stage is listed here; knockout fixtures are generated dynamically by
    the simulator from group results.
    """
    groups = _team_groups()
    rows = []
    match_id = 1
    base_date = pd.Timestamp("2026-06-11")
    for gi, group in enumerate(_GROUP_NAMES):
        teams = groups.loc[groups["group"] == group, "team"].tolist()
        # All 6 unique pairings in a 4-team round robin.
        pairings = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]
        for j, (a, b) in enumerate(pairings):
            date = (base_date + pd.Timedelta(days=gi % 6 + j * 2)).strftime("%Y-%m-%d")
            rows.append({
                "match_id": match_id,
                "stage": "group",
                "group": group,
                "date": date,
                "team_a": teams[a],
                "team_b": teams[b],
                "venue": f"Stadium {group}{j+1}",
                "neutral": True,  # World Cup group games are at neutral venues.
            })
            match_id += 1
    return pd.DataFrame(rows)


def write_all_templates() -> None:
    """Write every template CSV into data/raw/ (only missing files)."""
    config.ensure_dirs()
    writers = {
        config.GROUPS_FILE: make_groups_template,
        config.ELO_FILE: make_elo_template,
        config.FIFA_RANKINGS_FILE: make_fifa_rankings_template,
        config.RESULTS_FILE: make_results_template,
        config.FIXTURES_FILE: make_fixtures_template,
    }
    for path, fn in writers.items():
        if not path.exists():
            fn().to_csv(path, index=False)
            print(f"[templates] Created example file: {path.relative_to(config.PROJECT_ROOT)}")

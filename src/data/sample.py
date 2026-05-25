"""Synthetic SAMPLE data generator (opt-in demo only).

The project is CSV-first: real data goes in ``data/raw/`` (see ``templates`` /
``--mode validate-data``). This module is an explicit, opt-in generator
(``--mode sample-data``) that fabricates a *self-consistent* dataset so the
pipeline can be exercised end to end without real data.

Unlike a naive random dataset, the synthetic results here embed the same
effects the new features are designed to capture, so player/team-context and
team-dynamics features carry genuine signal:

* squad market value tracks team strength,
* injuries/suspensions temporarily weaken a team,
* recent form (momentum/streak) gives a temporary strength bump.

It reuses the team list and base strengths from the existing
``elo_ratings.csv`` so names stay consistent with the groups/fixtures files.
The data is NOT real and must be replaced with genuine sources for credible
forecasts.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .. import config
from . import templates

# How strongly each effect feeds into the synthetic match strengths (Elo pts).
# These are deliberately sizeable so the time-varying effects (which the static
# Elo/FIFA snapshots in this demo CANNOT capture) are detectable - i.e. so the
# new availability/momentum features can demonstrably add signal. On real data
# the magnitudes are whatever the data implies; here they are illustrative.
INJURY_ELO = 55.0       # per injured/suspended player
MOMENTUM_ELO = 150.0    # scale of the in-form bump
HOME_ADVANTAGE = 60.0

_TOURNAMENTS = ["Friendly", "FIFA World Cup", "UEFA Euro", "Copa America",
                "World Cup qualification", "Nations League"]
_T_WEIGHTS = np.array([0.45, 0.08, 0.07, 0.07, 0.23, 0.10])
_POSITIONS = ("GK", "DF", "DF", "MF", "MF", "FW", "FW", "FW")


def _team_strengths() -> dict[str, float]:
    """Read the 48 teams and their base strength from elo_ratings.csv."""
    elo = pd.read_csv(config.ELO_FILE)
    # One strength per team (latest if multiple snapshots).
    s = elo.sort_values("date").groupby("team")["elo_rating"].last()
    return {t: float(v) for t, v in s.items()}


def make_team_context(strengths: dict[str, float],
                      start: str = "2003-01-01", end: str = "2026-04-01",
                      seed: int = 11) -> pd.DataFrame:
    """Quarterly per-team context snapshots (injuries, value, xG)."""
    rng = np.random.default_rng(seed)
    snapshots = pd.date_range(start, end, freq="91D")
    rows = []
    for team, elo in strengths.items():
        # Market value scales with strength (stronger squads are worth more).
        base_value_m = max(25.0, (elo - 1400.0) * 8.0)
        for d in snapshots:
            # Occasional injury "crises" give the availability signal variance.
            injured = int(np.clip(rng.poisson(1.1) + (2 if rng.random() < 0.1 else 0), 0, 7))
            suspended = int(rng.binomial(2, 0.10))
            value = (base_value_m + rng.normal(0, 30)) * 1e6
            xg_for = max(0.3, 1.30 + (elo - 1500) / 250.0 + rng.normal(0, 0.15))
            xg_against = max(0.3, 1.30 - (elo - 1500) / 250.0 + rng.normal(0, 0.15))
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "team": team,
                "injured_players": injured,
                "suspended_players": suspended,
                "squad_market_value_eur": round(max(5e6, value), 0),
                "xg_for_10": round(xg_for, 3),
                "xg_against_10": round(xg_against, 3),
            })
    return pd.DataFrame(rows)


def _player_rows(team: str) -> list[dict]:
    """Small synthetic player pool for optional player-level feature demos."""
    return [
        {
            "player": f"{team} Player {i + 1}",
            "position": _POSITIONS[i % len(_POSITIONS)],
            "club": f"{team} FC",
            "is_probable_starter": i < 5,
        }
        for i in range(8)
    ]


def make_player_status(context: pd.DataFrame) -> pd.DataFrame:
    """Synthetic player availability snapshots aligned to team_context."""
    rows = []
    for r in context.itertuples(index=False):
        players = _player_rows(r.team)
        injured = int(getattr(r, "injured_players"))
        suspended = int(getattr(r, "suspended_players"))
        unavailable = injured + suspended
        for i, player in enumerate(players):
            if i < injured:
                status = "injured"
                injury_type = "sample soft-tissue"
            elif i < unavailable:
                status = "suspended"
                injury_type = ""
            elif i == unavailable and unavailable > 0:
                status = "doubtful"
                injury_type = "sample fitness test"
            else:
                status = "available"
                injury_type = ""
            rows.append({
                "as_of_date": r.date,
                "team": r.team,
                "player": player["player"],
                "position": player["position"],
                "club": player["club"],
                "squad_status": "provisional",
                "availability_status": status,
                "injury_type": injury_type,
                "expected_return": "",
                "is_probable_starter": player["is_probable_starter"],
                "source_url": "synthetic://sample-data",
            })
    return pd.DataFrame(rows)


def make_player_form(strengths: dict[str, float],
                     start_year: int = 2003, end_year: int = 2026,
                     seed: int = 21) -> pd.DataFrame:
    """Synthetic player-form rows with strength-correlated minutes/xG."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in range(start_year, end_year + 1):
        for team, elo in strengths.items():
            strength = max(0.1, (elo - 1350.0) / 600.0)
            for i, player in enumerate(_player_rows(team)):
                starter = player["is_probable_starter"]
                minutes = rng.integers(1800, 3600) if starter else rng.integers(400, 1800)
                attacking_role = 1.0 if player["position"] == "FW" else 0.45
                goals = max(0.0, rng.normal(5.0 * strength * attacking_role, 1.5))
                assists = max(0.0, rng.normal(3.5 * strength * attacking_role, 1.0))
                xg = max(0.0, goals + rng.normal(0.2, 0.8))
                xa = max(0.0, assists + rng.normal(0.2, 0.6))
                cards = max(0.0, rng.normal(4.0 if player["position"] != "FW" else 2.0, 1.5))
                rows.append({
                    "date": f"{year}-06-01",
                    "season": f"{year - 1}-{year}",
                    "team": team,
                    "player": player["player"],
                    "club": player["club"],
                    "competition": "Sample club season",
                    "minutes": int(minutes),
                    "starts": int(minutes // 90),
                    "goals": round(goals, 2),
                    "assists": round(assists, 2),
                    "xg": round(xg, 2),
                    "xa": round(xa, 2),
                    "cards": round(cards, 2),
                    "source_url": "synthetic://sample-data",
                })
    return pd.DataFrame(rows)


def make_team_status(strengths: dict[str, float],
                     start: str = "2003-01-01", end: str = "2026-04-01",
                     seed: int = 31) -> pd.DataFrame:
    """Synthetic team experience/stability snapshots."""
    rng = np.random.default_rng(seed)
    snapshots = pd.date_range(start, end, freq="182D")
    confeds = ["AFC", "CAF", "CONCACAF", "CONMEBOL", "UEFA"]
    rows = []
    for i, (team, elo) in enumerate(strengths.items()):
        confed = confeds[i % len(confeds)]
        for d in snapshots:
            rows.append({
                "as_of_date": d.strftime("%Y-%m-%d"),
                "team": team,
                "average_age": round(np.clip(26.0 + rng.normal(0, 1.8), 21, 33), 2),
                "total_caps": int(max(80, (elo - 1300) * 1.8 + rng.normal(0, 35))),
                "coach_tenure_days": int(max(30, rng.gamma(3.0, 220.0))),
                "fifa_confederation": confed,
                "source_url": "synthetic://sample-data",
            })
    return pd.DataFrame(rows)


def make_match_context(seed: int = 41) -> pd.DataFrame:
    """Synthetic fixture weather/travel context for the 2026 fixtures."""
    if not config.FIXTURES_FILE.exists():
        return templates.make_template(templates.optional_spec("match_context.csv"))
    rng = np.random.default_rng(seed)
    fixtures = pd.read_csv(config.FIXTURES_FILE)
    rows = []
    for m in fixtures.itertuples(index=False):
        travel_a = float(rng.integers(300, 5000))
        travel_b = float(rng.integers(300, 5000))
        rows.append({
            "match_id": m.match_id,
            "date": m.date,
            "team_a": m.team_a,
            "team_b": m.team_b,
            "venue": m.venue,
            "city": f"Sample City {m.group}",
            "temperature_c": round(float(rng.normal(24, 5)), 1),
            "humidity_pct": round(float(np.clip(rng.normal(55, 15), 15, 95)), 1),
            "wind_kmh": round(float(np.clip(rng.normal(12, 6), 0, 45)), 1),
            "altitude_m": round(float(np.clip(rng.normal(350, 500), 0, 2200)), 0),
            "team_a_travel_km": travel_a,
            "team_b_travel_km": travel_b,
            "source_url": "synthetic://sample-data",
        })
    return pd.DataFrame(rows)


def _context_index(context: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-team (sorted snapshot dates, absences) arrays for fast as-of lookup."""
    idx: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ctx = context.copy()
    ctx["date"] = pd.to_datetime(ctx["date"])
    ctx["absences"] = ctx["injured_players"] + ctx["suspended_players"]
    for team, sub in ctx.sort_values("date").groupby("team"):
        idx[team] = (sub["date"].to_numpy(), sub["absences"].to_numpy())
    return idx


def _absences_asof(team: str, date: pd.Timestamp,
                   idx: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    dates, vals = idx.get(team, (None, None))
    if dates is None:
        return 0.0
    pos = int(np.searchsorted(dates, np.datetime64(date), side="right")) - 1
    return float(vals[pos]) if pos >= 0 else 0.0


def make_results(strengths: dict[str, float], context: pd.DataFrame,
                 n_matches: int = 2200, start: str = "2003-01-01",
                 end: str = "2026-04-01", seed: int = 7) -> pd.DataFrame:
    """Synthetic match history with injury + momentum effects baked in."""
    rng = np.random.default_rng(seed)
    teams = list(strengths.keys())
    dates = pd.date_range(start, end, periods=n_matches)
    ctx_idx = _context_index(context)
    form5: dict[str, deque] = defaultdict(lambda: deque(maxlen=5))

    def momentum_bump(team: str) -> float:
        pts = form5[team]
        if not pts:
            return 0.0
        win_rate = sum(1 for p in pts if p == 3) / len(pts)
        return (win_rate - 0.4) * MOMENTUM_ELO

    def eff_strength(team: str, date, home: bool) -> float:
        s = strengths[team]
        s -= _absences_asof(team, date, ctx_idx) * INJURY_ELO
        s += momentum_bump(team)
        if home:
            s += HOME_ADVANTAGE
        return s

    rows = []
    for i in range(n_matches):
        home, away = rng.choice(teams, size=2, replace=False)
        date = dates[i]
        neutral = bool(rng.random() < 0.3)
        sh = eff_strength(home, date, home=not neutral)
        sa = eff_strength(away, date, home=False)
        diff = sh - sa
        lam_home = max(0.15, 1.35 * np.exp(diff / 600))
        lam_away = max(0.15, 1.35 * np.exp(-diff / 600))
        hs, as_ = int(rng.poisson(lam_home)), int(rng.poisson(lam_away))
        # Update momentum trackers.
        for t, gf, ga in ((home, hs, as_), (away, as_, hs)):
            form5[t].append(3 if gf > ga else (1 if gf == ga else 0))
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "home_team": home, "away_team": away,
            "home_score": hs, "away_score": as_,
            "tournament": str(rng.choice(_TOURNAMENTS, p=_T_WEIGHTS)),
            "neutral": neutral,
            "country": "Neutral" if neutral else home,
        })
    return pd.DataFrame(rows)


def write_sample_data() -> None:
    """Regenerate results + write a populated team_context.csv into data/raw/.

    Requires the existing strength files (elo_ratings.csv) and leaves the
    groups/fixtures/fifa files untouched so team names stay consistent.
    """
    config.ensure_dirs()
    if not config.ELO_FILE.exists():
        raise FileNotFoundError(
            "elo_ratings.csv is required to seed sample data. Run "
            "`python main.py --mode validate-data` and populate it first."
        )
    strengths = _team_strengths()
    context = make_team_context(strengths)
    player_status = make_player_status(context)
    player_form = make_player_form(strengths)
    team_status = make_team_status(strengths)
    match_context = make_match_context()
    results = make_results(strengths, context)

    context_path = config.RAW_DIR / "team_context.csv"
    player_status_path = config.RAW_DIR / "player_status.csv"
    player_form_path = config.RAW_DIR / "player_form.csv"
    team_status_path = config.RAW_DIR / "team_status.csv"
    match_context_path = config.RAW_DIR / "match_context.csv"
    context.to_csv(context_path, index=False)
    player_status.to_csv(player_status_path, index=False)
    player_form.to_csv(player_form_path, index=False)
    team_status.to_csv(team_status_path, index=False)
    match_context.to_csv(match_context_path, index=False)
    results.to_csv(config.RESULTS_FILE, index=False)
    print(f"[sample-data] Wrote synthetic {results.shape[0]} results -> "
          f"{_rel(config.RESULTS_FILE)}")
    print(f"[sample-data] Wrote synthetic team context -> {_rel(context_path)}")
    print(f"[sample-data] Wrote synthetic player status -> {_rel(player_status_path)}")
    print(f"[sample-data] Wrote synthetic player form -> {_rel(player_form_path)}")
    print(f"[sample-data] Wrote synthetic team status -> {_rel(team_status_path)}")
    print(f"[sample-data] Wrote synthetic match context -> {_rel(match_context_path)}")
    print("[sample-data] NOTE: synthetic data for demos only - not real.")


def _rel(path) -> str:
    """Project-relative path string when possible, else the full path."""
    try:
        return str(path.relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path)

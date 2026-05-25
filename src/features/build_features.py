"""Match-level feature engineering.

Two entry points:

* :func:`build_training_features` turns the historical results table into a
  supervised dataset (features + ``result`` target) for the logistic model.
* :func:`build_fixture_features` builds the same feature columns for the 2026
  fixtures (or any hypothetical matchup) using the latest ratings and the most
  recent form available, so the trained model can predict them.

The two share the same column layout so a model trained on one can score the
other without surprises.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable

import numpy as np
import pandas as pd

from .. import config
from ..data import loaders

# Feature columns consumed by the models (order matters for the design matrix).
NUMERIC_FEATURES = [
    "elo_a", "elo_b", "elo_diff",
    "fifa_rank_a", "fifa_rank_b", "fifa_rank_diff",
    "fifa_points_a", "fifa_points_b", "fifa_points_diff",
    "team_a_recent_win_rate_10", "team_b_recent_win_rate_10", "recent_win_rate_diff",
    "team_a_recent_goals_for_10", "team_b_recent_goals_for_10",
    "team_a_recent_goals_against_10", "team_b_recent_goals_against_10",
    "recent_goal_diff_a", "recent_goal_diff_b", "recent_goal_diff_diff",
    "team_a_rest_days", "team_b_rest_days", "rest_days_diff",
    "team_a_absences", "team_b_absences", "absence_diff",
    "team_a_market_value", "team_b_market_value", "market_value_diff",
    "team_a_xg_diff_10", "team_b_xg_diff_10", "xg_diff_delta",
    "neutral", "is_world_cup", "is_major_tournament",
]

ROLL_WINDOW = 10  # number of recent matches used for form features

_WORLD_CUP_KEYS = ("fifa world cup", "world cup")
_MAJOR_KEYS = (
    "world cup", "euro", "copa america", "nations league",
    "african cup", "asian cup", "gold cup", "confederations",
)


def _is_world_cup(tournament: str) -> int:
    t = str(tournament).lower()
    # Exclude qualifiers, which contain "qualification".
    if "qualif" in t:
        return 0
    return int(any(k in t for k in _WORLD_CUP_KEYS))


def _is_major(tournament: str) -> int:
    t = str(tournament).lower()
    if "qualif" in t:
        return 0
    return int(any(k in t for k in _MAJOR_KEYS))


def _result_from_scores(home_score: int, away_score: int) -> int:
    """Encode result from team_a (home) perspective: 0 loss, 1 draw, 2 win."""
    if home_score > away_score:
        return 2
    if home_score == away_score:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Rolling form tracker
# ---------------------------------------------------------------------------
class _FormTracker:
    """Maintain each team's last ``window`` matches to derive form features.

    State is updated *after* a match is scored, so features for a match only
    use information available before kickoff (no leakage).
    """

    def __init__(self, window: int = ROLL_WINDOW):
        self.window = window
        self._gf: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._ga: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._pts: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._last_date: dict[str, pd.Timestamp] = {}

    def features(self, team: str, date: pd.Timestamp | None = None) -> dict[str, float]:
        gf = self._gf[team]
        ga = self._ga[team]
        pts = self._pts[team]
        n = len(pts)
        rest_days = 7.0
        if date is not None and team in self._last_date:
            rest_days = float(np.clip((date - self._last_date[team]).days, 0, 30))
        if n == 0:
            # Neutral priors for teams with no history yet.
            return {
                "win_rate": 0.33,
                "gf": 1.2,
                "ga": 1.2,
                "gd": 0.0,
                "rest_days": rest_days,
            }
        win_rate = sum(1 for p in pts if p == 3) / n
        gf_mean = float(np.mean(gf))
        ga_mean = float(np.mean(ga))
        return {"win_rate": win_rate, "gf": gf_mean, "ga": ga_mean,
                "gd": gf_mean - ga_mean, "rest_days": rest_days}

    def update(self, team: str, goals_for: int, goals_against: int,
               date: pd.Timestamp | None = None) -> None:
        self._gf[team].append(goals_for)
        self._ga[team].append(goals_against)
        if goals_for > goals_against:
            self._pts[team].append(3)
        elif goals_for == goals_against:
            self._pts[team].append(1)
        else:
            self._pts[team].append(0)
        if date is not None:
            self._last_date[team] = date


# ---------------------------------------------------------------------------
# Ratings attachment (as-of join with fallback)
# ---------------------------------------------------------------------------
def _ratings_lookup() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return time-sorted elo and fifa frames for as-of joins."""
    elo = loaders.load_elo().sort_values("date")[["date", "team", "elo_rating"]]
    fifa = loaders.load_fifa_rankings().sort_values("date")[
        ["date", "team", "fifa_rank", "fifa_points"]
    ]
    return elo, fifa


def _team_context_lookup() -> pd.DataFrame:
    """Return optional team context snapshots sorted for as-of lookups."""
    context = loaders.load_team_context()
    if context.empty:
        return context
    return context.sort_values("date")


def _asof_for_team(team: str, date: pd.Timestamp, sorted_df: pd.DataFrame,
                   value_cols: list[str], fallbacks: dict[str, float]) -> dict:
    """Most recent rating for ``team`` on/before ``date``; fallback if none.

    Works whether the ratings file is a single snapshot (template) or a full
    time series (real data). If no row precedes the date we fall back to the
    team's earliest known value, then to the supplied default.
    """
    sub = sorted_df[sorted_df["team"] == team]
    out = {}
    if sub.empty:
        return dict(fallbacks)
    prior = sub[sub["date"] <= date]
    row = (prior.iloc[-1] if not prior.empty else sub.iloc[0])
    for c in value_cols:
        val = row[c]
        out[c] = float(val) if pd.notna(val) else fallbacks[c]
    return out


def _context_for_team(team: str, date: pd.Timestamp | None,
                      context_df: pd.DataFrame,
                      form: dict[str, float]) -> dict[str, float]:
    """Return optional context features, with neutral defaults when absent."""
    def num(value, default: float = 0.0) -> float:
        return default if pd.isna(value) else float(value)

    defaults = {
        "absences": 0.0,
        "market_value": 0.0,
        "xg_diff_10": form["gd"],
    }
    if context_df.empty:
        return defaults

    sub = context_df[context_df["team"] == team]
    if sub.empty:
        return defaults
    if date is None:
        row = sub.iloc[-1]
    else:
        prior = sub[sub["date"] <= date]
        row = prior.iloc[-1] if not prior.empty else sub.iloc[0]

    injured = num(row.get("injured_players", 0.0))
    suspended = num(row.get("suspended_players", 0.0))
    market_value = num(row.get("squad_market_value_eur", 0.0))
    xg_for = row.get("xg_for_10", np.nan)
    xg_against = row.get("xg_against_10", np.nan)
    if pd.isna(xg_for) or pd.isna(xg_against):
        xg_diff = form["gd"]
    else:
        xg_diff = float(xg_for) - float(xg_against)
    return {
        "absences": injured + suspended,
        "market_value": market_value,
        "xg_diff_10": xg_diff,
    }


def build_training_features(results: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the supervised training table from historical results.

    team_a is the home team and team_b the away team. The ``result`` target is
    encoded from team_a's perspective (0 loss / 1 draw / 2 win).
    """
    if results is None:
        results = loaders.load_results()
    results = results.sort_values("date").reset_index(drop=True)

    elo_df, fifa_df = _ratings_lookup()
    context_df = _team_context_lookup()
    default_elo = config.DEFAULT_ELO
    default_rank = float(fifa_df["fifa_rank"].max()) if not fifa_df.empty else 200.0
    default_pts = float(fifa_df["fifa_points"].min()) if not fifa_df.empty else 0.0

    tracker = _FormTracker()
    rows = []
    for r in results.itertuples(index=False):
        a, b = r.home_team, r.away_team
        date = r.date

        ea = _asof_for_team(a, date, elo_df, ["elo_rating"], {"elo_rating": default_elo})
        eb = _asof_for_team(b, date, elo_df, ["elo_rating"], {"elo_rating": default_elo})
        fa = _asof_for_team(a, date, fifa_df, ["fifa_rank", "fifa_points"],
                            {"fifa_rank": default_rank, "fifa_points": default_pts})
        fb = _asof_for_team(b, date, fifa_df, ["fifa_rank", "fifa_points"],
                            {"fifa_rank": default_rank, "fifa_points": default_pts})

        form_a = tracker.features(a, date)
        form_b = tracker.features(b, date)
        context_a = _context_for_team(a, date, context_df, form_a)
        context_b = _context_for_team(b, date, context_df, form_b)

        rows.append(_assemble_row(
            a, b,
            ea["elo_rating"], eb["elo_rating"],
            fa["fifa_rank"], fb["fifa_rank"],
            fa["fifa_points"], fb["fifa_points"],
            form_a, form_b, context_a, context_b,
            neutral=int(bool(r.neutral)),
            is_wc=_is_world_cup(r.tournament),
            is_major=_is_major(r.tournament),
            result=_result_from_scores(r.home_score, r.away_score),
        ))

        # Update form state after recording features.
        tracker.update(a, r.home_score, r.away_score, date)
        tracker.update(b, r.away_score, r.home_score, date)

    return pd.DataFrame(rows)


def _assemble_row(team_a, team_b, elo_a, elo_b, rank_a, rank_b,
                  pts_a, pts_b, form_a, form_b, context_a=None, context_b=None,
                  neutral=1, is_wc=1, is_major=1, result=None) -> dict:
    """Build one feature dict shared by training and prediction paths."""
    context_a = context_a or {
        "absences": 0.0,
        "market_value": 0.0,
        "xg_diff_10": form_a["gd"],
    }
    context_b = context_b or {
        "absences": 0.0,
        "market_value": 0.0,
        "xg_diff_10": form_b["gd"],
    }
    row = {
        "team_a": team_a, "team_b": team_b,
        "elo_a": elo_a, "elo_b": elo_b, "elo_diff": elo_a - elo_b,
        "fifa_rank_a": rank_a, "fifa_rank_b": rank_b,
        "fifa_rank_diff": rank_a - rank_b,
        "fifa_points_a": pts_a, "fifa_points_b": pts_b,
        "fifa_points_diff": pts_a - pts_b,
        "team_a_recent_win_rate_10": form_a["win_rate"],
        "team_b_recent_win_rate_10": form_b["win_rate"],
        "recent_win_rate_diff": form_a["win_rate"] - form_b["win_rate"],
        "team_a_recent_goals_for_10": form_a["gf"],
        "team_b_recent_goals_for_10": form_b["gf"],
        "team_a_recent_goals_against_10": form_a["ga"],
        "team_b_recent_goals_against_10": form_b["ga"],
        "recent_goal_diff_a": form_a["gd"],
        "recent_goal_diff_b": form_b["gd"],
        "recent_goal_diff_diff": form_a["gd"] - form_b["gd"],
        "team_a_rest_days": form_a["rest_days"],
        "team_b_rest_days": form_b["rest_days"],
        "rest_days_diff": form_a["rest_days"] - form_b["rest_days"],
        "team_a_absences": context_a["absences"],
        "team_b_absences": context_b["absences"],
        "absence_diff": context_b["absences"] - context_a["absences"],
        "team_a_market_value": context_a["market_value"],
        "team_b_market_value": context_b["market_value"],
        "market_value_diff": context_a["market_value"] - context_b["market_value"],
        "team_a_xg_diff_10": context_a["xg_diff_10"],
        "team_b_xg_diff_10": context_b["xg_diff_10"],
        "xg_diff_delta": context_a["xg_diff_10"] - context_b["xg_diff_10"],
        "neutral": int(neutral),
        "is_world_cup": int(is_wc),
        "is_major_tournament": int(is_major),
    }
    if result is not None:
        row["result"] = result
    return row


def current_form_table(results: pd.DataFrame | None = None) -> dict[str, dict]:
    """Return the latest rolling-form features per team (end of history).

    Used to build features for 2026 fixtures, which have no match history of
    their own yet.
    """
    if results is None:
        results = loaders.load_results()
    results = results.sort_values("date").reset_index(drop=True)
    tracker = _FormTracker()
    for r in results.itertuples(index=False):
        tracker.update(r.home_team, r.home_score, r.away_score, r.date)
        tracker.update(r.away_team, r.away_score, r.home_score, r.date)
    teams = set(results["home_team"]) | set(results["away_team"])
    return {t: tracker.features(t) for t in teams}


def build_fixture_features(team_a: str, team_b: str, neutral: bool,
                           ratings: pd.DataFrame, form: dict[str, dict],
                           context: pd.DataFrame | None = None,
                           match_date: pd.Timestamp | str | None = None,
                           is_world_cup: int = 1,
                           is_major_tournament: int = 1) -> dict:
    """Build a single feature row for a 2026 matchup.

    ``ratings`` is the one-row-per-team frame from ``loaders.latest_ratings``;
    ``form`` is the dict from :func:`current_form_table`.
    """
    rmap = ratings.set_index("team")
    default_rank = float(ratings["fifa_rank"].max())
    default_pts = float(ratings["fifa_points"].min())

    def get(team):
        if team in rmap.index:
            row = rmap.loc[team]
            return float(row["elo_rating"]), float(row["fifa_rank"]), float(row["fifa_points"])
        return config.DEFAULT_ELO, default_rank, default_pts

    elo_a, rank_a, pts_a = get(team_a)
    elo_b, rank_b, pts_b = get(team_b)
    neutral_prior = {
        "win_rate": 0.33,
        "gf": 1.2,
        "ga": 1.2,
        "gd": 0.0,
        "rest_days": 7.0,
    }
    form_a = form.get(team_a, neutral_prior)
    form_b = form.get(team_b, neutral_prior)
    context_df = context if context is not None else _team_context_lookup()
    date = pd.Timestamp(match_date) if match_date is not None else None
    context_a = _context_for_team(team_a, date, context_df, form_a)
    context_b = _context_for_team(team_b, date, context_df, form_b)

    return _assemble_row(
        team_a, team_b, elo_a, elo_b, rank_a, rank_b, pts_a, pts_b,
        form_a, form_b, context_a, context_b, neutral=int(neutral),
        is_wc=is_world_cup, is_major=is_major_tournament,
    )


def features_to_matrix(rows: Iterable[dict] | pd.DataFrame) -> pd.DataFrame:
    """Select and order the numeric feature columns into a model matrix."""
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    return df[NUMERIC_FEATURES].astype(float)

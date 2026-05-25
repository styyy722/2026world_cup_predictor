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

    def features(self, team: str) -> dict[str, float]:
        gf = self._gf[team]
        ga = self._ga[team]
        pts = self._pts[team]
        n = len(pts)
        if n == 0:
            # Neutral priors for teams with no history yet.
            return {"win_rate": 0.33, "gf": 1.2, "ga": 1.2, "gd": 0.0}
        win_rate = sum(1 for p in pts if p == 3) / n
        gf_mean = float(np.mean(gf))
        ga_mean = float(np.mean(ga))
        return {"win_rate": win_rate, "gf": gf_mean, "ga": ga_mean,
                "gd": gf_mean - ga_mean}

    def update(self, team: str, goals_for: int, goals_against: int) -> None:
        self._gf[team].append(goals_for)
        self._ga[team].append(goals_against)
        if goals_for > goals_against:
            self._pts[team].append(3)
        elif goals_for == goals_against:
            self._pts[team].append(1)
        else:
            self._pts[team].append(0)


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


def build_training_features(results: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the supervised training table from historical results.

    team_a is the home team and team_b the away team. The ``result`` target is
    encoded from team_a's perspective (0 loss / 1 draw / 2 win).
    """
    if results is None:
        results = loaders.load_results()
    results = results.sort_values("date").reset_index(drop=True)

    elo_df, fifa_df = _ratings_lookup()
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

        form_a = tracker.features(a)
        form_b = tracker.features(b)

        rows.append(_assemble_row(
            a, b,
            ea["elo_rating"], eb["elo_rating"],
            fa["fifa_rank"], fb["fifa_rank"],
            fa["fifa_points"], fb["fifa_points"],
            form_a, form_b,
            neutral=int(bool(r.neutral)),
            is_wc=_is_world_cup(r.tournament),
            is_major=_is_major(r.tournament),
            result=_result_from_scores(r.home_score, r.away_score),
        ))

        # Update form state after recording features.
        tracker.update(a, r.home_score, r.away_score)
        tracker.update(b, r.away_score, r.home_score)

    return pd.DataFrame(rows)


def _assemble_row(team_a, team_b, elo_a, elo_b, rank_a, rank_b,
                  pts_a, pts_b, form_a, form_b, neutral, is_wc, is_major,
                  result=None) -> dict:
    """Build one feature dict shared by training and prediction paths."""
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
        tracker.update(r.home_team, r.home_score, r.away_score)
        tracker.update(r.away_team, r.away_score, r.home_score)
    teams = set(results["home_team"]) | set(results["away_team"])
    return {t: tracker.features(t) for t in teams}


def build_fixture_features(team_a: str, team_b: str, neutral: bool,
                           ratings: pd.DataFrame, form: dict[str, dict],
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
    neutral_prior = {"win_rate": 0.33, "gf": 1.2, "ga": 1.2, "gd": 0.0}
    form_a = form.get(team_a, neutral_prior)
    form_b = form.get(team_b, neutral_prior)

    return _assemble_row(
        team_a, team_b, elo_a, elo_b, rank_a, rank_b, pts_a, pts_b,
        form_a, form_b, neutral=int(neutral),
        is_wc=is_world_cup, is_major=is_major_tournament,
    )


def features_to_matrix(rows: Iterable[dict] | pd.DataFrame) -> pd.DataFrame:
    """Select and order the numeric feature columns into a model matrix."""
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    return df[NUMERIC_FEATURES].astype(float)

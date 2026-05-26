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
    "team_a_momentum", "team_b_momentum", "momentum_diff",
    "team_a_streak", "team_b_streak", "streak_diff",
    "team_a_absences", "team_b_absences", "absence_diff",
    "team_a_unavailable_players", "team_b_unavailable_players",
    "unavailable_players_diff",
    "team_a_doubtful_players", "team_b_doubtful_players", "doubtful_players_diff",
    "team_a_probable_starters", "team_b_probable_starters", "probable_starters_diff",
    "team_a_availability_score", "team_b_availability_score", "availability_score_diff",
    "team_a_market_value", "team_b_market_value", "market_value_diff",
    "team_a_xg_diff_10", "team_b_xg_diff_10", "xg_diff_delta",
    "team_a_player_minutes_index", "team_b_player_minutes_index", "player_minutes_index_diff",
    "team_a_goal_contribution_90", "team_b_goal_contribution_90",
    "goal_contribution_90_diff",
    "team_a_player_xg_xa_90", "team_b_player_xg_xa_90", "player_xg_xa_90_diff",
    "team_a_cards_per_90", "team_b_cards_per_90", "cards_per_90_diff",
    "team_a_average_age", "team_b_average_age", "average_age_diff",
    "team_a_total_caps", "team_b_total_caps", "total_caps_diff",
    "team_a_coach_tenure_days", "team_b_coach_tenure_days", "coach_tenure_diff",
    "temperature_c", "humidity_pct", "wind_kmh", "altitude_m",
    "team_a_travel_km", "team_b_travel_km", "travel_km_diff",
    "neutral", "is_world_cup", "is_major_tournament",
]

ROLL_WINDOW = 10  # number of recent matches used for form features
SHORT_WINDOW = 5  # short window for recent-form momentum
STREAK_CLIP = 5   # cap the signed win/loss streak

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


def _is_world_cup_match(tournament: str) -> bool:
    return bool(_is_world_cup(tournament))


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


def filter_training_results(
    results: pd.DataFrame,
    lookback_years: int = config.TRAINING_LOOKBACK_YEARS,
    last_world_cup_year: int = config.LAST_WORLD_CUP_YEAR,
) -> pd.DataFrame:
    """Return the modelling subset: recent history plus the last World Cup.

    Non-World-Cup matches are kept only when they fall inside the most recent
    ``lookback_years`` of the available results. FIFA World Cup proper matches
    are limited to ``last_world_cup_year`` so older tournament editions do not
    teach the model from long-retired squads.
    """
    if results.empty:
        return results.copy()
    df = results.sort_values("date").reset_index(drop=True).copy()
    cutoff = df["date"].max() - pd.DateOffset(years=lookback_years)
    is_wc = df["tournament"].apply(_is_world_cup_match)
    recent_non_wc = (df["date"] >= cutoff) & ~is_wc
    last_wc = is_wc & (df["date"].dt.year == last_world_cup_year)
    return df.loc[recent_non_wc | last_wc].reset_index(drop=True)


def training_window_summary(results: pd.DataFrame) -> dict[str, object]:
    """Summarise the configured training-result subset for logs/docs."""
    if results.empty:
        return {
            "total_matches": 0,
            "kept_matches": 0,
            "cutoff_date": None,
            "latest_date": None,
            "last_world_cup_year": config.LAST_WORLD_CUP_YEAR,
        }
    latest = results["date"].max()
    cutoff = latest - pd.DateOffset(years=config.TRAINING_LOOKBACK_YEARS)
    kept = filter_training_results(results)
    return {
        "total_matches": int(len(results)),
        "kept_matches": int(len(kept)),
        "cutoff_date": cutoff,
        "latest_date": latest,
        "last_world_cup_year": config.LAST_WORLD_CUP_YEAR,
    }


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
        self._streak: dict[str, int] = defaultdict(int)

    def features(self, team: str, date: pd.Timestamp | None = None) -> dict[str, float]:
        gf = self._gf[team]
        ga = self._ga[team]
        pts = self._pts[team]
        n = len(pts)
        rest_days = 7.0
        if date is not None and team in self._last_date:
            rest_days = float(np.clip((date - self._last_date[team]).days, 0, 30))
        streak = float(self._streak[team])
        if n == 0:
            # Neutral priors for teams with no history yet.
            return {
                "win_rate": 0.33,
                "form5": 0.33,
                "momentum": 0.0,
                "streak": 0.0,
                "gf": 1.2,
                "ga": 1.2,
                "gd": 0.0,
                "rest_days": rest_days,
            }
        win_rate = sum(1 for p in pts if p == 3) / n
        recent = list(pts)[-SHORT_WINDOW:]
        form5 = sum(1 for p in recent if p == 3) / len(recent)
        gf_mean = float(np.mean(gf))
        ga_mean = float(np.mean(ga))
        return {"win_rate": win_rate, "form5": form5,
                # Momentum: recent (last 5) win rate vs the longer baseline.
                "momentum": form5 - win_rate, "streak": streak,
                "gf": gf_mean, "ga": ga_mean,
                "gd": gf_mean - ga_mean, "rest_days": rest_days}

    def update(self, team: str, goals_for: int, goals_against: int,
               date: pd.Timestamp | None = None) -> None:
        self._gf[team].append(goals_for)
        self._ga[team].append(goals_against)
        if goals_for > goals_against:
            self._pts[team].append(3)
            self._streak[team] = max(1, self._streak[team] + 1)
        elif goals_for == goals_against:
            self._pts[team].append(1)
            self._streak[team] = 0  # a draw resets the win/loss run
        else:
            self._pts[team].append(0)
            self._streak[team] = min(-1, self._streak[team] - 1)
        self._streak[team] = int(np.clip(self._streak[team], -STREAK_CLIP, STREAK_CLIP))
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


def _optional_sources() -> dict[str, pd.DataFrame]:
    """Load optional player/team/match context frames once per feature build."""
    return {
        "player_status": loaders.load_player_status(),
        "player_form": loaders.load_player_form(),
        "team_status": loaders.load_team_status(),
        "match_context": loaders.load_match_context(),
    }


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
                      form: dict[str, float],
                      optional: dict[str, pd.DataFrame] | None = None
                      ) -> dict[str, float]:
    """Return optional context features, with neutral defaults when absent."""
    def num(value, default: float = 0.0) -> float:
        return default if pd.isna(value) else float(value)

    defaults = {
        "absences": 0.0,
        "unavailable_players": 0.0,
        "doubtful_players": 0.0,
        "probable_starters": 0.0,
        "availability_score": 0.0,
        "market_value": 0.0,
        "xg_diff_10": form["gd"],
        "player_minutes_index": 0.0,
        "goal_contribution_90": 0.0,
        "player_xg_xa_90": 0.0,
        "cards_per_90": 0.0,
        "average_age": 27.0,
        "total_caps": 0.0,
        "coach_tenure_days": 0.0,
    }
    optional = optional or {}

    player_status = _player_status_features(
        team, date, optional.get("player_status", pd.DataFrame())
    )
    player_form = _player_form_features(
        team, date, optional.get("player_form", pd.DataFrame())
    )
    team_status = _team_status_features(
        team, date, optional.get("team_status", pd.DataFrame())
    )

    if context_df.empty:
        return {**defaults, **player_status, **player_form, **team_status}

    sub = context_df[context_df["team"] == team]
    if sub.empty:
        return {**defaults, **player_status, **player_form, **team_status}
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
    base = {
        "absences": injured + suspended,
        "market_value": market_value,
        "xg_diff_10": xg_diff,
    }
    return {**defaults, **base, **player_status, **player_form, **team_status}


def _latest_rows_by_entity(df: pd.DataFrame, team: str, date_col: str,
                           date: pd.Timestamp | None,
                           entity_col: str = "player") -> pd.DataFrame:
    """Latest rows for a team, optionally as-of a date, grouped by entity."""
    if df.empty or "team" not in df.columns:
        return pd.DataFrame(columns=df.columns)
    sub = df[df["team"] == team].copy()
    if sub.empty:
        return sub
    if date is not None and date_col in sub.columns:
        sub = sub[sub[date_col] <= date]
    if sub.empty:
        return sub
    if date_col in sub.columns:
        sub = sub.sort_values(date_col)
    if entity_col in sub.columns:
        return sub.groupby(entity_col, as_index=False).last()
    return sub.tail(1)


def _is_truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "starter"}


def _player_status_features(team: str, date: pd.Timestamp | None,
                            status_df: pd.DataFrame) -> dict[str, float]:
    """Aggregate player availability rows into team-level features."""
    rows = _latest_rows_by_entity(status_df, team, "as_of_date", date)
    if rows.empty:
        return {
            "unavailable_players": 0.0,
            "doubtful_players": 0.0,
            "probable_starters": 0.0,
            "availability_score": 0.0,
        }
    status = rows.get("availability_status", pd.Series("", index=rows.index))
    status = status.fillna("").astype(str).str.lower()
    unavailable = status.str.contains("injured|suspended|out|unavailable", regex=True)
    doubtful = status.str.contains("doubt|questionable|test", regex=True)
    if "is_probable_starter" in rows.columns:
        starters = rows["is_probable_starter"].apply(_is_truthy)
    else:
        squad = rows.get("squad_status", pd.Series("", index=rows.index))
        starters = squad.fillna("").astype(str).str.lower().str.contains("starter")
    probable_starters = float(starters.sum())
    unavailable_count = float(unavailable.sum())
    doubtful_count = float(doubtful.sum())
    return {
        "unavailable_players": unavailable_count,
        "doubtful_players": doubtful_count,
        "probable_starters": probable_starters,
        "availability_score": probable_starters - unavailable_count - 0.5 * doubtful_count,
    }


def _player_form_features(team: str, date: pd.Timestamp | None,
                          form_df: pd.DataFrame) -> dict[str, float]:
    """Aggregate player performance rows into team-level form features."""
    rows = _latest_rows_by_entity(form_df, team, "date", date)
    if rows.empty:
        return {
            "player_minutes_index": 0.0,
            "goal_contribution_90": 0.0,
            "player_xg_xa_90": 0.0,
            "cards_per_90": 0.0,
        }

    def col(name: str) -> pd.Series:
        if name not in rows.columns:
            return pd.Series(0.0, index=rows.index)
        return pd.to_numeric(rows[name], errors="coerce").fillna(0.0)

    minutes = col("minutes")
    total_minutes = float(minutes.sum())
    denom = max(total_minutes, 1.0)
    goals = col("goals")
    assists = col("assists")
    xg = col("xg")
    xa = col("xa")
    cards = col("cards")
    return {
        "player_minutes_index": total_minutes / 10_000.0,
        "goal_contribution_90": float((goals.sum() + assists.sum()) * 90.0 / denom),
        "player_xg_xa_90": float((xg.sum() + xa.sum()) * 90.0 / denom),
        "cards_per_90": float(cards.sum() * 90.0 / denom),
    }


def _team_status_features(team: str, date: pd.Timestamp | None,
                          status_df: pd.DataFrame) -> dict[str, float]:
    """Return team experience/stability features from latest team snapshot."""
    row = _latest_rows_by_entity(status_df, team, "as_of_date", date, entity_col="team")
    if row.empty:
        return {"average_age": 27.0, "total_caps": 0.0, "coach_tenure_days": 0.0}
    r = row.iloc[-1]

    def num(name: str, default: float = 0.0) -> float:
        value = r.get(name, default)
        return default if pd.isna(value) else float(value)

    return {
        "average_age": num("average_age", 27.0),
        "total_caps": num("total_caps", 0.0),
        "coach_tenure_days": num("coach_tenure_days", 0.0),
    }


def _match_context_features(match_id: int | str | None,
                            team_a: str | None,
                            team_b: str | None,
                            match_context_df: pd.DataFrame | None
                            ) -> dict[str, float]:
    """Return fixture context, with neutral defaults if no row is available."""
    defaults = {
        "temperature_c": 20.0,
        "humidity_pct": 50.0,
        "wind_kmh": 10.0,
        "altitude_m": 0.0,
        "team_a_travel_km": 0.0,
        "team_b_travel_km": 0.0,
        "travel_km_diff": 0.0,
    }
    if match_context_df is None or match_context_df.empty:
        return defaults
    ctx = match_context_df
    row = pd.DataFrame()
    if match_id is not None and "match_id" in ctx.columns:
        row = ctx[ctx["match_id"].astype(str) == str(match_id)]
    if row.empty and team_a is not None and team_b is not None:
        direct = (ctx["team_a"] == team_a) & (ctx["team_b"] == team_b)
        reverse = (ctx["team_a"] == team_b) & (ctx["team_b"] == team_a)
        row = ctx[direct | reverse]
    if row.empty:
        return defaults
    r = row.iloc[-1]

    def num(name: str, default: float) -> float:
        value = r.get(name, default)
        return default if pd.isna(value) else float(value)

    travel_a = num("team_a_travel_km", 0.0)
    travel_b = num("team_b_travel_km", 0.0)
    if str(r.get("team_a", team_a)) == team_b and str(r.get("team_b", team_b)) == team_a:
        travel_a, travel_b = travel_b, travel_a
    return {
        "temperature_c": num("temperature_c", 20.0),
        "humidity_pct": num("humidity_pct", 50.0),
        "wind_kmh": num("wind_kmh", 10.0),
        "altitude_m": num("altitude_m", 0.0),
        "team_a_travel_km": travel_a,
        "team_b_travel_km": travel_b,
        "travel_km_diff": travel_a - travel_b,
    }


def build_training_features(
    results: pd.DataFrame | None = None,
    *,
    apply_training_window: bool = True,
) -> pd.DataFrame:
    """Build the supervised training table from historical results.

    team_a is the home team and team_b the away team. The ``result`` target is
    encoded from team_a's perspective (0 loss / 1 draw / 2 win).
    """
    if results is None:
        results = loaders.load_results()
    results = results.sort_values("date").reset_index(drop=True)
    if apply_training_window:
        results = filter_training_results(results)

    elo_df, fifa_df = _ratings_lookup()
    context_df = _team_context_lookup()
    optional = _optional_sources()
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
        context_a = _context_for_team(a, date, context_df, form_a, optional)
        context_b = _context_for_team(b, date, context_df, form_b, optional)
        match_context = _match_context_features(None, a, b, optional["match_context"])

        rows.append(_assemble_row(
            a, b,
            ea["elo_rating"], eb["elo_rating"],
            fa["fifa_rank"], fb["fifa_rank"],
            fa["fifa_points"], fb["fifa_points"],
            form_a, form_b, context_a, context_b, match_context,
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
                  match_context=None, neutral=1, is_wc=1, is_major=1,
                  result=None) -> dict:
    """Build one feature dict shared by training and prediction paths."""
    context_a = context_a or {
        "absences": 0.0,
        "unavailable_players": 0.0,
        "doubtful_players": 0.0,
        "probable_starters": 0.0,
        "availability_score": 0.0,
        "market_value": 0.0,
        "xg_diff_10": form_a["gd"],
        "player_minutes_index": 0.0,
        "goal_contribution_90": 0.0,
        "player_xg_xa_90": 0.0,
        "cards_per_90": 0.0,
        "average_age": 27.0,
        "total_caps": 0.0,
        "coach_tenure_days": 0.0,
    }
    context_b = context_b or {
        "absences": 0.0,
        "unavailable_players": 0.0,
        "doubtful_players": 0.0,
        "probable_starters": 0.0,
        "availability_score": 0.0,
        "market_value": 0.0,
        "xg_diff_10": form_b["gd"],
        "player_minutes_index": 0.0,
        "goal_contribution_90": 0.0,
        "player_xg_xa_90": 0.0,
        "cards_per_90": 0.0,
        "average_age": 27.0,
        "total_caps": 0.0,
        "coach_tenure_days": 0.0,
    }
    match_context = match_context or _match_context_features(None, None, None, None)
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
        "team_a_momentum": form_a["momentum"],
        "team_b_momentum": form_b["momentum"],
        "momentum_diff": form_a["momentum"] - form_b["momentum"],
        "team_a_streak": form_a["streak"],
        "team_b_streak": form_b["streak"],
        "streak_diff": form_a["streak"] - form_b["streak"],
        "team_a_absences": context_a["absences"],
        "team_b_absences": context_b["absences"],
        "absence_diff": context_b["absences"] - context_a["absences"],
        "team_a_unavailable_players": context_a["unavailable_players"],
        "team_b_unavailable_players": context_b["unavailable_players"],
        "unavailable_players_diff": (
            context_b["unavailable_players"] - context_a["unavailable_players"]
        ),
        "team_a_doubtful_players": context_a["doubtful_players"],
        "team_b_doubtful_players": context_b["doubtful_players"],
        "doubtful_players_diff": context_b["doubtful_players"] - context_a["doubtful_players"],
        "team_a_probable_starters": context_a["probable_starters"],
        "team_b_probable_starters": context_b["probable_starters"],
        "probable_starters_diff": (
            context_a["probable_starters"] - context_b["probable_starters"]
        ),
        "team_a_availability_score": context_a["availability_score"],
        "team_b_availability_score": context_b["availability_score"],
        "availability_score_diff": (
            context_a["availability_score"] - context_b["availability_score"]
        ),
        "team_a_market_value": context_a["market_value"],
        "team_b_market_value": context_b["market_value"],
        "market_value_diff": context_a["market_value"] - context_b["market_value"],
        "team_a_xg_diff_10": context_a["xg_diff_10"],
        "team_b_xg_diff_10": context_b["xg_diff_10"],
        "xg_diff_delta": context_a["xg_diff_10"] - context_b["xg_diff_10"],
        "team_a_player_minutes_index": context_a["player_minutes_index"],
        "team_b_player_minutes_index": context_b["player_minutes_index"],
        "player_minutes_index_diff": (
            context_a["player_minutes_index"] - context_b["player_minutes_index"]
        ),
        "team_a_goal_contribution_90": context_a["goal_contribution_90"],
        "team_b_goal_contribution_90": context_b["goal_contribution_90"],
        "goal_contribution_90_diff": (
            context_a["goal_contribution_90"] - context_b["goal_contribution_90"]
        ),
        "team_a_player_xg_xa_90": context_a["player_xg_xa_90"],
        "team_b_player_xg_xa_90": context_b["player_xg_xa_90"],
        "player_xg_xa_90_diff": context_a["player_xg_xa_90"] - context_b["player_xg_xa_90"],
        "team_a_cards_per_90": context_a["cards_per_90"],
        "team_b_cards_per_90": context_b["cards_per_90"],
        "cards_per_90_diff": context_a["cards_per_90"] - context_b["cards_per_90"],
        "team_a_average_age": context_a["average_age"],
        "team_b_average_age": context_b["average_age"],
        "average_age_diff": context_a["average_age"] - context_b["average_age"],
        "team_a_total_caps": context_a["total_caps"],
        "team_b_total_caps": context_b["total_caps"],
        "total_caps_diff": context_a["total_caps"] - context_b["total_caps"],
        "team_a_coach_tenure_days": context_a["coach_tenure_days"],
        "team_b_coach_tenure_days": context_b["coach_tenure_days"],
        "coach_tenure_diff": (
            context_a["coach_tenure_days"] - context_b["coach_tenure_days"]
        ),
        **match_context,
        "neutral": int(neutral),
        "is_world_cup": int(is_wc),
        "is_major_tournament": int(is_major),
    }
    if result is not None:
        row["result"] = result
    return row


def current_form_table(
    results: pd.DataFrame | None = None,
    *,
    apply_training_window: bool = True,
) -> dict[str, dict]:
    """Return the latest rolling-form features per team (end of history).

    Used to build features for 2026 fixtures, which have no match history of
    their own yet.
    """
    if results is None:
        results = loaders.load_results()
    results = results.sort_values("date").reset_index(drop=True)
    if apply_training_window:
        results = filter_training_results(results)
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
                           match_id: int | str | None = None,
                           optional: dict[str, pd.DataFrame] | None = None,
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
        "form5": 0.33,
        "momentum": 0.0,
        "streak": 0.0,
        "gf": 1.2,
        "ga": 1.2,
        "gd": 0.0,
        "rest_days": 7.0,
    }
    form_a = form.get(team_a, neutral_prior)
    form_b = form.get(team_b, neutral_prior)
    context_df = context if context is not None else _team_context_lookup()
    optional = optional or _optional_sources()
    date = pd.Timestamp(match_date) if match_date is not None else None
    context_a = _context_for_team(team_a, date, context_df, form_a, optional)
    context_b = _context_for_team(team_b, date, context_df, form_b, optional)
    match_context = _match_context_features(
        match_id, team_a, team_b, optional.get("match_context", pd.DataFrame())
    )

    return _assemble_row(
        team_a, team_b, elo_a, elo_b, rank_a, rank_b, pts_a, pts_b,
        form_a, form_b, context_a, context_b, match_context,
        neutral=int(neutral),
        is_wc=is_world_cup, is_major=is_major_tournament,
    )


def features_to_matrix(rows: Iterable[dict] | pd.DataFrame) -> pd.DataFrame:
    """Select and order the numeric feature columns into a model matrix."""
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    return df[NUMERIC_FEATURES].astype(float)

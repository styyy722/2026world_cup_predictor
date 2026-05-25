"""Unified match predictor used by the tournament simulator.

Wraps either the logistic model or the Elo model behind a single interface:

    proba(team_a, team_b, neutral) -> (P_team_a_win, P_draw, P_team_b_win)

For the logistic model we cache per-team ratings and form so feature building
is cheap inside the hot simulation loop. We also expose each team's Elo, which
the knockout penalty-shootout tie-break uses for relative strength.
"""
from __future__ import annotations

import pandas as pd

from .. import config
from ..features import build_features as bf
from ..models import common
from ..models.elo_model import EloModel

# Model kinds that are sklearn-compatible classifiers driven by features.
_CLASSIFIER_KINDS = ("logistic", "xgboost", "lightgbm", "catboost", "tree")


class MatchPredictor:
    """Predict WDL probabilities for any team pairing.

    Works with the Elo baseline or any feature-driven classifier (the logistic
    baseline or a gradient-boosted tree backend), since all classifiers share
    the ``models.common`` probability interface.
    """

    def __init__(self, model_kind: str, model=None,
                 ratings: pd.DataFrame | None = None,
                 form: dict[str, dict] | None = None,
                 context: pd.DataFrame | None = None,
                 odds: pd.DataFrame | None = None,
                 odds_weight: float = 0.0):
        self.model_kind = model_kind
        self.model = model
        self.ratings = ratings
        self.form = form or {}
        self.context = context
        self.odds = odds if odds is not None and not odds.empty else None
        self.odds_weight = max(0.0, min(1.0, float(odds_weight)))
        # Map team -> Elo for fast lookup (used by Elo model and tie-breaks).
        self.elo_by_team: dict[str, float] = {}
        if ratings is not None:
            self.elo_by_team = dict(zip(ratings["team"], ratings["elo_rating"]))
        if model_kind == "elo":
            self.model = self.model or EloModel(self.elo_by_team)
        # Cache deterministic per-matchup probabilities: ratings and form are
        # fixed during a run, so (team_a, team_b, neutral) -> probs is stable.
        # This keeps the Monte Carlo loop fast (thousands of repeated matchups).
        self._cache: dict[tuple, tuple[float, float, float]] = {}

    @classmethod
    def from_logistic(cls, model, ratings: pd.DataFrame,
                      form: dict[str, dict],
                      context: pd.DataFrame | None = None,
                      odds: pd.DataFrame | None = None,
                      odds_weight: float = 0.0) -> "MatchPredictor":
        return cls(
            "logistic", model=model, ratings=ratings, form=form,
            context=context, odds=odds, odds_weight=odds_weight,
        )

    @classmethod
    def from_classifier(cls, model_kind: str, model, ratings: pd.DataFrame,
                        form: dict[str, dict],
                        context: pd.DataFrame | None = None,
                        odds: pd.DataFrame | None = None,
                        odds_weight: float = 0.0) -> "MatchPredictor":
        """Build a predictor for any feature-driven classifier backend."""
        return cls(
            model_kind, model=model, ratings=ratings, form=form,
            context=context, odds=odds, odds_weight=odds_weight,
        )

    @classmethod
    def from_elo(cls, ratings: pd.DataFrame,
                 odds: pd.DataFrame | None = None,
                 odds_weight: float = 0.0) -> "MatchPredictor":
        return cls("elo", ratings=ratings, odds=odds, odds_weight=odds_weight)

    def elo(self, team: str) -> float:
        return self.elo_by_team.get(team, config.DEFAULT_ELO)

    def proba(self, team_a: str, team_b: str,
              neutral: bool = True,
              match_id: int | str | None = None,
              match_date=None) -> tuple[float, float, float]:
        """Return (P_team_a_win, P_draw, P_team_b_win)."""
        key = (team_a, team_b, bool(neutral), match_id, str(match_date))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if self.model_kind == "elo":
            result = self.model.proba(team_a, team_b, neutral)
        else:
            row = bf.build_fixture_features(
                team_a, team_b, neutral, self.ratings, self.form,
                context=self.context, match_date=match_date,
                is_world_cup=1, is_major_tournament=1,
            )
            p = common.predict_proba_dicts(self.model, row)
            result = (p["team_a_win"], p["draw"], p["team_b_win"])

        odds_probs = self._odds_proba(team_a, team_b, match_id=match_id)
        if odds_probs is not None and self.odds_weight > 0:
            w = self.odds_weight
            result = tuple(
                (1 - w) * model_p + w * odds_p
                for model_p, odds_p in zip(result, odds_probs)
            )

        self._cache[key] = result
        return result

    def _odds_proba(self, team_a: str, team_b: str,
                    match_id: int | str | None = None
                    ) -> tuple[float, float, float] | None:
        """Return normalised bookmaker-implied WDL probabilities if available."""
        if self.odds is None:
            return None
        odds = self.odds
        match = pd.DataFrame()
        if match_id is not None and "match_id" in odds.columns:
            match = odds[odds["match_id"].astype(str) == str(match_id)]
        if match.empty:
            direct = (odds["team_a"] == team_a) & (odds["team_b"] == team_b)
            reverse = (odds["team_a"] == team_b) & (odds["team_b"] == team_a)
            match = odds[direct | reverse]
        if match.empty:
            return None

        row = match.iloc[-1]
        dec = [
            float(row["team_a_decimal_odds"]),
            float(row["draw_decimal_odds"]),
            float(row["team_b_decimal_odds"]),
        ]
        if any(pd.isna(v) or v <= 1.0 for v in dec):
            return None
        implied = [1.0 / v for v in dec]
        total = sum(implied)
        probs = tuple(v / total for v in implied)
        if row["team_a"] == team_b and row["team_b"] == team_a:
            return (probs[2], probs[1], probs[0])
        return probs

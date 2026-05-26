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
from ..models import market_odds
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
                 optional_context: dict[str, pd.DataFrame] | None = None,
                 odds: pd.DataFrame | None = None,
                 odds_weight: float = 0.0,
                 odds_method: str = "shin",
                 odds_blend: str = "logarithmic"):
        self.model_kind = model_kind
        self.model = model
        self.ratings = ratings
        self.form = form or {}
        self.context = context
        self.optional_context = optional_context or {}
        self.odds = odds if odds is not None and not odds.empty else None
        self.odds_weight = max(0.0, min(1.0, float(odds_weight)))
        self.odds_method = odds_method
        self.odds_blend = odds_blend
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
                      optional_context: dict[str, pd.DataFrame] | None = None,
                      odds: pd.DataFrame | None = None,
                      odds_weight: float = 0.0,
                      odds_method: str = "shin",
                      odds_blend: str = "logarithmic") -> "MatchPredictor":
        return cls(
            "logistic", model=model, ratings=ratings, form=form,
            context=context, optional_context=optional_context,
            odds=odds, odds_weight=odds_weight,
            odds_method=odds_method, odds_blend=odds_blend,
        )

    @classmethod
    def from_classifier(cls, model_kind: str, model, ratings: pd.DataFrame,
                        form: dict[str, dict],
                        context: pd.DataFrame | None = None,
                        optional_context: dict[str, pd.DataFrame] | None = None,
                        odds: pd.DataFrame | None = None,
                        odds_weight: float = 0.0,
                        odds_method: str = "shin",
                        odds_blend: str = "logarithmic") -> "MatchPredictor":
        """Build a predictor for any feature-driven classifier backend."""
        return cls(
            model_kind, model=model, ratings=ratings, form=form,
            context=context, optional_context=optional_context,
            odds=odds, odds_weight=odds_weight,
            odds_method=odds_method, odds_blend=odds_blend,
        )

    @classmethod
    def from_elo(cls, ratings: pd.DataFrame,
                 odds: pd.DataFrame | None = None,
                 odds_weight: float = 0.0,
                 odds_method: str = "shin",
                 odds_blend: str = "logarithmic") -> "MatchPredictor":
        return cls(
            "elo", ratings=ratings, odds=odds, odds_weight=odds_weight,
            odds_method=odds_method, odds_blend=odds_blend,
        )

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
                context=self.context, match_date=match_date, match_id=match_id,
                optional=self.optional_context,
                is_world_cup=1, is_major_tournament=1,
            )
            p = common.predict_proba_dicts(self.model, row)
            result = (p["team_a_win"], p["draw"], p["team_b_win"])

        odds_probs = self._odds_proba(team_a, team_b, match_id=match_id)
        if odds_probs is not None and self.odds_weight > 0:
            result = market_odds.blend_probabilities(
                result,
                odds_probs,
                weight=self.odds_weight,
                method=self.odds_blend,
            )

        self._cache[key] = result
        return result

    def _odds_proba(self, team_a: str, team_b: str,
                    match_id: int | str | None = None
                    ) -> tuple[float, float, float] | None:
        """Return consensus no-vig betting-market WDL probabilities."""
        return market_odds.consensus_market_probabilities(
            self.odds,
            team_a,
            team_b,
            match_id=match_id,
            method=self.odds_method,
        )

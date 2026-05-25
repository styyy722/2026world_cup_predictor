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
                 form: dict[str, dict] | None = None):
        self.model_kind = model_kind
        self.model = model
        self.ratings = ratings
        self.form = form or {}
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
                      form: dict[str, dict]) -> "MatchPredictor":
        return cls("logistic", model=model, ratings=ratings, form=form)

    @classmethod
    def from_classifier(cls, model_kind: str, model, ratings: pd.DataFrame,
                        form: dict[str, dict]) -> "MatchPredictor":
        """Build a predictor for any feature-driven classifier backend."""
        return cls(model_kind, model=model, ratings=ratings, form=form)

    @classmethod
    def from_elo(cls, ratings: pd.DataFrame) -> "MatchPredictor":
        return cls("elo", ratings=ratings)

    def elo(self, team: str) -> float:
        return self.elo_by_team.get(team, config.DEFAULT_ELO)

    def proba(self, team_a: str, team_b: str,
              neutral: bool = True) -> tuple[float, float, float]:
        """Return (P_team_a_win, P_draw, P_team_b_win)."""
        key = (team_a, team_b, bool(neutral))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if self.model_kind == "elo":
            result = self.model.proba(team_a, team_b, neutral)
        else:
            row = bf.build_fixture_features(
                team_a, team_b, neutral, self.ratings, self.form,
                is_world_cup=1, is_major_tournament=1,
            )
            p = common.predict_proba_dicts(self.model, row)
            result = (p["team_a_win"], p["draw"], p["team_b_win"])

        self._cache[key] = result
        return result

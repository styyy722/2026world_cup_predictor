"""Elo-based baseline match model.

A deliberately simple, parameter-light baseline to compare against the
logistic regression model. It maps the Elo difference into win/draw/loss
probabilities using the standard Elo expected-score formula plus a tunable
draw model.
"""
from __future__ import annotations

import numpy as np

from .. import config

# Standard Elo scale: a 400-point gap means ~10x expected-score odds.
ELO_SCALE = 400.0

# Home advantage in Elo points applied to team_a when the match is not neutral.
HOME_ADVANTAGE = 65.0

# Controls how much probability mass goes to draws. Larger -> more draws.
DRAW_BASE = 0.28
DRAW_DECAY = 600.0  # draws decay as the Elo gap widens


def expected_score(elo_a: float, elo_b: float) -> float:
    """Elo expected score for team_a (between 0 and 1).

    This is the classic logistic expectation: 1 / (1 + 10^(-diff/400)).
    A value of 0.5 means evenly matched.
    """
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / ELO_SCALE))


def convert_elo_to_wdl_probabilities(elo_a: float, elo_b: float,
                                     neutral: bool = True) -> tuple[float, float, float]:
    """Convert an Elo matchup into (P_win_a, P_draw, P_win_b).

    The draw probability shrinks as the absolute Elo gap grows; the remaining
    mass is split between the two teams in proportion to the Elo expected
    score. Probabilities sum to 1.
    """
    eff_a = elo_a + (0.0 if neutral else HOME_ADVANTAGE)
    exp_a = expected_score(eff_a, elo_b)

    gap = abs(eff_a - elo_b)
    p_draw = DRAW_BASE * np.exp(-gap / DRAW_DECAY)
    p_draw = float(np.clip(p_draw, 0.05, 0.45))

    remaining = 1.0 - p_draw
    p_win_a = remaining * exp_a
    p_win_b = remaining * (1.0 - exp_a)
    return p_win_a, p_draw, p_win_b


def predict_match(elo_a: float, elo_b: float,
                  neutral: bool = True) -> dict[str, float]:
    """Predict a single match, returning a labelled probability dict."""
    pa, pd_, pb = convert_elo_to_wdl_probabilities(elo_a, elo_b, neutral)
    return {"team_a_win": pa, "draw": pd_, "team_b_win": pb}


class EloModel:
    """Thin wrapper exposing a predict_proba compatible with the simulator.

    It looks up each team's Elo from a ``{team: elo}`` mapping so the simulator
    can call it the same way it calls the logistic model.
    """

    def __init__(self, elo_by_team: dict[str, float] | None = None):
        self.elo_by_team = elo_by_team or {}

    def proba(self, team_a: str, team_b: str, neutral: bool = True) -> tuple[float, float, float]:
        ea = self.elo_by_team.get(team_a, config.DEFAULT_ELO)
        eb = self.elo_by_team.get(team_b, config.DEFAULT_ELO)
        return convert_elo_to_wdl_probabilities(ea, eb, neutral)

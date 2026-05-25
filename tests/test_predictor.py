"""Tests for predictor extensions such as betting-odds blending."""
import pandas as pd

from src.simulation.predictor import MatchPredictor


def test_elo_predictor_blends_optional_betting_odds():
    ratings = pd.DataFrame({
        "team": ["A", "B"],
        "elo_rating": [1500.0, 1500.0],
    })
    odds = pd.DataFrame({
        "match_id": [1],
        "team_a": ["A"],
        "team_b": ["B"],
        "team_a_decimal_odds": [2.0],
        "draw_decimal_odds": [4.0],
        "team_b_decimal_odds": [4.0],
    })

    predictor = MatchPredictor.from_elo(ratings, odds=odds, odds_weight=1.0)
    pa, pd_, pb = predictor.proba("A", "B", match_id=1)

    assert round(pa, 6) == 0.5
    assert round(pd_, 6) == 0.25
    assert round(pb, 6) == 0.25

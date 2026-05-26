"""Tests for no-vig betting-market probability utilities."""
import numpy as np
import pandas as pd

from src.models import market_odds


def test_basic_no_vig_probabilities_remove_overround():
    probs = market_odds.basic_no_vig_probabilities([1.80, 3.50, 5.00])

    assert np.isclose(sum(probs), 1.0)
    assert probs[0] > probs[1] > probs[2]


def test_shin_no_vig_probabilities_are_valid():
    probs = market_odds.shin_no_vig_probabilities([1.80, 3.50, 5.00])

    assert np.isclose(sum(probs), 1.0)
    assert all(0.0 < p < 1.0 for p in probs)


def test_consensus_market_probabilities_aggregate_and_reorient():
    odds = pd.DataFrame({
        "match_id": [1, 1],
        "team_a": ["A", "B"],
        "team_b": ["B", "A"],
        "team_a_decimal_odds": [1.80, 5.20],
        "draw_decimal_odds": [3.50, 3.40],
        "team_b_decimal_odds": [5.00, 1.90],
        "bookmaker": ["Book 1", "Book 2"],
    })

    probs = market_odds.consensus_market_probabilities(
        odds, "A", "B", match_id=1, method="basic"
    )
    first = market_odds.basic_no_vig_probabilities([1.80, 3.50, 5.00])
    second = market_odds.basic_no_vig_probabilities([5.20, 3.40, 1.90])
    expected = np.median(np.asarray([first, (second[2], second[1], second[0])]), axis=0)
    expected = expected / expected.sum()

    assert probs is not None
    assert np.allclose(probs, expected)


def test_consensus_prefers_closing_latest_snapshot_per_bookmaker():
    odds = pd.DataFrame({
        "match_id": [1, 1, 1],
        "snapshot_time": ["2026-06-01T10:00:00", "2026-06-01T18:00:00", "2026-06-01T12:00:00"],
        "team_a": ["A", "A", "A"],
        "team_b": ["B", "B", "B"],
        "team_a_decimal_odds": [4.00, 1.80, 1.90],
        "draw_decimal_odds": [3.00, 3.50, 3.40],
        "team_b_decimal_odds": [2.00, 5.00, 4.80],
        "bookmaker": ["Book 1", "Book 1", "Book 2"],
        "is_closing": [False, True, True],
    })

    probs = market_odds.consensus_market_probabilities(
        odds, "A", "B", match_id=1, method="basic"
    )
    first = market_odds.basic_no_vig_probabilities([1.80, 3.50, 5.00])
    second = market_odds.basic_no_vig_probabilities([1.90, 3.40, 4.80])
    expected = np.median(np.asarray([first, second]), axis=0)
    expected = expected / expected.sum()

    assert probs is not None
    assert np.allclose(probs, expected)


def test_logarithmic_blend_preserves_probability_vector():
    blended = market_odds.blend_probabilities(
        (0.50, 0.25, 0.25),
        (0.25, 0.25, 0.50),
        weight=0.5,
        method="logarithmic",
    )

    assert np.isclose(sum(blended), 1.0)
    assert np.isclose(blended[0], blended[2])
    assert blended[0] > blended[1]

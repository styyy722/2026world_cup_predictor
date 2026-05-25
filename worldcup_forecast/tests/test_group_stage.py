"""Tests for group ranking and third-placed selection."""
import numpy as np

from src.simulation import simulate_group_stage as gs


def test_rank_group_orders_by_points_then_gd_then_gf():
    rng = np.random.default_rng(0)
    stats = {
        "A": {"points": 9, "gf": 5, "ga": 1, "gd": 4, "played": 3},
        "B": {"points": 6, "gf": 7, "ga": 4, "gd": 3, "played": 3},
        "C": {"points": 6, "gf": 4, "ga": 4, "gd": 0, "played": 3},
        "D": {"points": 0, "gf": 1, "ga": 8, "gd": -7, "played": 3},
    }
    order = gs.rank_group(stats, rng)
    assert order[0] == "A"          # most points
    assert order[1] == "B"          # tie on points, better GD
    assert order[2] == "C"
    assert order[3] == "D"


def test_rank_group_uses_goals_for_tiebreak():
    rng = np.random.default_rng(0)
    stats = {
        "X": {"points": 6, "gf": 8, "ga": 4, "gd": 4, "played": 3},
        "Y": {"points": 6, "gf": 6, "ga": 2, "gd": 4, "played": 3},
    }
    order = gs.rank_group(stats, rng)
    assert order[0] == "X"  # same points & GD, more goals for


def test_select_best_third_placed_picks_top_n():
    rng = np.random.default_rng(1)
    thirds = [
        {"team": f"T{i}", "points": i, "gf": i, "ga": 0, "gd": i}
        for i in range(12)
    ]
    best = gs.select_best_third_placed(thirds, rng, n=8)
    assert len(best) == 8
    selected_points = {t["points"] for t in best}
    # The 8 highest-point teams (points 4..11) should be selected.
    assert selected_points == set(range(4, 12))

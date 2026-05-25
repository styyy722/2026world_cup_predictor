"""Tests for knockout match/bracket simulation."""
from itertools import combinations

import numpy as np

from src.simulation import simulate_match as sm
from src.simulation import simulate_knockout_stage as ko


class _FakePredictor:
    """Deterministic predictor: team_a always wins in regular time."""

    def __init__(self, elos):
        self._elos = elos

    def elo(self, team):
        return self._elos.get(team, 1500.0)

    def proba(self, a, b, neutral=True):
        return (1.0, 0.0, 0.0)  # team_a certain win


def test_knockout_winner_returns_a_team():
    rng = np.random.default_rng(0)
    w = sm.simulate_knockout_winner("A", "B", (0.4, 0.2, 0.4), 1600, 1500, rng)
    assert w in ("A", "B")


def test_knockout_winner_certain_win():
    rng = np.random.default_rng(0)
    w = sm.simulate_knockout_winner("A", "B", (1.0, 0.0, 0.0), 1500, 1500, rng)
    assert w == "A"


def test_draw_resolved_to_a_team():
    rng = np.random.default_rng(0)
    # Pure draw probability -> must still resolve via shootout.
    w = sm.simulate_knockout_winner("A", "B", (0.0, 1.0, 0.0), 1500, 1500, rng)
    assert w in ("A", "B")


def test_bracket_size_and_full_run():
    elos = {f"T{i}": 1500 + i for i in range(32)}
    qualified = [{"team": f"T{i}"} for i in range(32)]
    pred = _FakePredictor(elos)
    bracket = ko.build_r32_bracket(qualified, pred)
    assert len(bracket) == 32
    assert len(set(bracket)) == 32  # all distinct

    rng = np.random.default_rng(0)
    res = ko.simulate_knockout(bracket, pred, rng)
    assert res["champion"] in bracket
    assert res["runner_up"] in bracket
    assert res["champion"] != res["runner_up"]
    assert len(res["semi_finalists"]) == 4


def test_official_r32_pairings_use_published_slots():
    standings = {}
    qualified_thirds = []
    for group in "ABCDEFGHIJKL":
        ranked = []
        for rank in range(1, 5):
            ranked.append({
                "team": f"{group}{rank}",
                "group": group,
                "rank": rank,
                "points": 12 - rank,
                "gd": 4 - rank,
                "gf": 5 - rank,
            })
        standings[group] = ranked
        if group in set("ABCDEFGH"):
            qualified_thirds.append(ranked[2])

    pairings = ko.r32_pairings_from_group_results({
        "standings": standings,
        "qualified_thirds": qualified_thirds,
    })

    by_match = {p["match_id"]: p for p in pairings}
    assert by_match[73]["team_a"] == "A2"
    assert by_match[73]["team_b"] == "B2"
    assert by_match[75]["team_a"] == "F1"
    assert by_match[75]["team_b"] == "C2"
    assert by_match[84]["team_a"] == "H1"
    assert by_match[84]["team_b"] == "J2"

    third_place_teams = [
        p["team_b"] for p in pairings if p["slot_b"].startswith("3")
    ]
    assert len(third_place_teams) == 8
    assert len(set(third_place_teams)) == 8


def test_third_place_slot_assignment_covers_all_group_combinations():
    third_slots = [slot for _, _, slot in ko.R32_SLOT_ORDER if slot.startswith("3")]
    for groups in combinations("ABCDEFGHIJKL", 8):
        qualified = [{"group": group} for group in groups]
        assigned = ko._assign_third_place_slots(qualified, third_slots)
        assert set(assigned.values()) == set(groups)

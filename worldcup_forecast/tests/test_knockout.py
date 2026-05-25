"""Tests for knockout match/bracket simulation."""
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

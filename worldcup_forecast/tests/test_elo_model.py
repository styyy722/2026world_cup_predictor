"""Tests for the Elo baseline model."""
import numpy as np

from src.models import elo_model as em


def test_expected_score_symmetry():
    # Equal Elo -> 0.5 expected score.
    assert abs(em.expected_score(1500, 1500) - 0.5) < 1e-9


def test_expected_score_monotonic():
    # Stronger team always has higher expected score.
    assert em.expected_score(1700, 1500) > 0.5
    assert em.expected_score(1300, 1500) < 0.5


def test_expected_score_400_gap():
    # The classic Elo property: a 400-point gap -> ~10x odds (~0.909).
    val = em.expected_score(1900, 1500)
    assert abs(val - (10 / 11)) < 1e-6


def test_wdl_probabilities_sum_to_one():
    for ea, eb in [(1500, 1500), (1800, 1400), (1450, 1750)]:
        pa, pd_, pb = em.convert_elo_to_wdl_probabilities(ea, eb)
        assert abs((pa + pd_ + pb) - 1.0) < 1e-9
        assert all(0 <= p <= 1 for p in (pa, pd_, pb))


def test_stronger_team_more_likely_to_win():
    pa, _pd, pb = em.convert_elo_to_wdl_probabilities(1800, 1400, neutral=True)
    assert pa > pb

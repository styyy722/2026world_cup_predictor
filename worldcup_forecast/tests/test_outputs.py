"""Tests that simulation outputs are well-formed probabilities."""
import numpy as np

from src.data import loaders
from src.simulation import simulate_tournament as sim
from src.simulation.predictor import MatchPredictor


def _small_sim(n=200):
    groups = loaders.load_groups()
    fixtures = loaders.load_fixtures()
    ratings = loaders.latest_ratings()
    predictor = MatchPredictor.from_elo(ratings)
    return sim.run_simulations(predictor, groups, fixtures, n_simulations=n)


def test_group_placement_probabilities_sum_to_at_most_one():
    res = _small_sim()
    ts = res["team_stage"]
    placement = ts["prob_group_1st"] + ts["prob_group_2nd"] + ts["prob_group_3rd"]
    # A team finishes 1st/2nd/3rd/4th, so these three sum to <= 1.
    assert (placement <= 1.0 + 1e-9).all()


def test_stage_probabilities_are_monotonic():
    res = _small_sim()
    ts = res["team_stage"]
    # Reaching a later stage can never be more likely than an earlier one.
    assert (ts["prob_reach_r16"] <= ts["prob_reach_r32"] + 1e-9).all()
    assert (ts["prob_reach_qf"] <= ts["prob_reach_r16"] + 1e-9).all()
    assert (ts["prob_reach_sf"] <= ts["prob_reach_qf"] + 1e-9).all()
    assert (ts["prob_reach_final"] <= ts["prob_reach_sf"] + 1e-9).all()
    assert (ts["prob_champion"] <= ts["prob_reach_final"] + 1e-9).all()


def test_all_probabilities_in_unit_interval():
    res = _small_sim()
    ts = res["team_stage"]
    prob_cols = [c for c in ts.columns if c.startswith("prob_")]
    for c in prob_cols:
        assert (ts[c] >= -1e-9).all() and (ts[c] <= 1.0 + 1e-9).all()


def test_champion_probabilities_sum_to_one():
    res = _small_sim()
    total = res["team_stage"]["prob_champion"].sum()
    # Exactly one champion per simulation -> probabilities sum to 1.
    assert abs(total - 1.0) < 1e-9


def test_r32_field_is_32_teams():
    # 24 group qualifiers + 8 best thirds = 32 reach R32 each simulation.
    res = _small_sim()
    expected = 32.0 / 48.0  # mean fraction of teams reaching R32 if uniform
    total_r32 = res["team_stage"]["prob_reach_r32"].sum()
    # Summed reach-R32 probability across teams equals the field size (32).
    assert abs(total_r32 - 32.0) < 1e-6

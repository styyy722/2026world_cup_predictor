"""Tests for team-dynamics features and the synthetic sample generator."""
import numpy as np
import pandas as pd

from src.features import build_features as bf


def test_new_dynamics_columns_present_in_matrix():
    feats = bf.build_training_features()
    X = bf.features_to_matrix(feats)
    for col in ["team_a_momentum", "team_b_momentum", "momentum_diff",
                "team_a_streak", "team_b_streak", "streak_diff"]:
        assert col in X.columns
    assert not X.isna().any().any()


def test_streak_tracks_consecutive_results():
    t = bf._FormTracker()
    # Three wins -> streak +3; a loss flips to -1; a draw resets to 0.
    for _ in range(3):
        t.update("X", 2, 0)
    assert t.features("X")["streak"] == 3.0
    t.update("X", 0, 1)
    assert t.features("X")["streak"] == -1.0
    t.update("X", 1, 1)
    assert t.features("X")["streak"] == 0.0


def test_streak_is_clipped():
    t = bf._FormTracker()
    for _ in range(20):
        t.update("Y", 3, 0)
    assert t.features("Y")["streak"] == float(bf.STREAK_CLIP)


def test_momentum_positive_when_recent_better_than_baseline():
    t = bf._FormTracker()
    # Build a 10-match window: first 5 losses, last 5 wins -> recent > baseline.
    for _ in range(5):
        t.update("Z", 0, 1)
    for _ in range(5):
        t.update("Z", 2, 0)
    f = t.features("Z")
    assert f["form5"] == 1.0          # last five all wins
    assert f["momentum"] > 0.0        # recent form above the 10-match rate


def test_sample_generator_writes_team_context(tmp_path, monkeypatch):
    from src import config
    from src.data import sample
    # Point the data dir at a temp location seeded with a tiny elo file.
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    monkeypatch.setattr(config, "RESULTS_FILE", tmp_path / "international_results.csv")
    monkeypatch.setattr(config, "ELO_FILE", tmp_path / "elo_ratings.csv")
    pd.DataFrame({
        "date": ["2024-01-01"] * 4,
        "team": ["A", "B", "C", "D"],
        "elo_rating": [1900, 1700, 1500, 1400],
    }).to_csv(config.ELO_FILE, index=False)

    sample.write_sample_data()
    ctx = pd.read_csv(tmp_path / "team_context.csv")
    res = pd.read_csv(config.RESULTS_FILE)
    assert set(["injured_players", "squad_market_value_eur", "xg_for_10"]).issubset(ctx.columns)
    assert len(res) > 0
    # Stronger team should carry a higher average squad market value.
    val = ctx.groupby("team")["squad_market_value_eur"].mean()
    assert val["A"] > val["D"]

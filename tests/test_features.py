"""Tests for team-dynamics features and the synthetic sample generator."""
import numpy as np
import pandas as pd

from src.features import build_features as bf


def test_new_dynamics_columns_present_in_matrix():
    feats = bf.build_training_features()
    X = bf.features_to_matrix(feats)
    for col in ["team_a_momentum", "team_b_momentum", "momentum_diff",
                "team_a_streak", "team_b_streak", "streak_diff",
                "team_a_unavailable_players", "team_b_unavailable_players",
                "team_a_player_minutes_index", "team_b_player_minutes_index",
                "team_a_average_age", "team_b_average_age",
                "temperature_c", "travel_km_diff"]:
        assert col in X.columns
    assert not X.isna().any().any()


def test_training_filter_uses_recent_matches_and_last_world_cup_only():
    rows = pd.DataFrame({
        "date": pd.to_datetime([
            "2018-06-15",
            "2021-10-01",
            "2022-11-22",
            "2023-07-01",
            "2024-03-01",
            "2026-04-01",
        ]),
        "home_team": ["A", "C", "E", "G", "I", "K"],
        "away_team": ["B", "D", "F", "H", "J", "L"],
        "home_score": [1, 1, 2, 0, 3, 1],
        "away_score": [0, 1, 1, 1, 2, 0],
        "tournament": [
            "FIFA World Cup",
            "Friendly",
            "FIFA World Cup",
            "FIFA World Cup",
            "Friendly",
            "World Cup qualification",
        ],
        "neutral": [True, False, True, True, False, False],
        "country": ["X"] * 6,
    })

    filtered = bf.filter_training_results(rows, lookback_years=4, last_world_cup_year=2022)

    assert filtered["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2022-11-22",
        "2024-03-01",
        "2026-04-01",
    ]
    assert set(filtered["tournament"]) == {
        "FIFA World Cup",
        "Friendly",
        "World Cup qualification",
    }


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
    player_status = pd.read_csv(tmp_path / "player_status.csv")
    player_form = pd.read_csv(tmp_path / "player_form.csv")
    team_status = pd.read_csv(tmp_path / "team_status.csv")
    match_context = pd.read_csv(tmp_path / "match_context.csv")
    res = pd.read_csv(config.RESULTS_FILE)
    assert set(["injured_players", "squad_market_value_eur", "xg_for_10"]).issubset(ctx.columns)
    assert set(["availability_status", "is_probable_starter"]).issubset(player_status.columns)
    assert set(["minutes", "xg", "xa"]).issubset(player_form.columns)
    assert set(["average_age", "coach_tenure_days"]).issubset(team_status.columns)
    assert set(["temperature_c", "team_a_travel_km"]).issubset(match_context.columns)
    assert len(res) > 0
    # Stronger team should carry a higher average squad market value.
    val = ctx.groupby("team")["squad_market_value_eur"].mean()
    assert val["A"] > val["D"]

"""Tests for walk-forward backtesting and hyperparameter tuning."""
import pytest

from src.evaluation import backtest as bt
from src.features import build_features as bf
from src.models import common


@pytest.fixture(scope="module")
def features():
    return bf.build_training_features()


def test_walk_forward_uses_all_history(features):
    res = bt.walk_forward_backtest(features, model_kind="logistic", n_splits=5)
    # With 5 expanding folds the test set covers the large majority of matches.
    assert res["n_test_total"] > 0.5 * len(features)
    assert 0.0 <= res["accuracy"] <= 1.0
    assert res["log_loss"] > 0.0
    assert 0.0 <= res["brier"] <= 2.0
    assert not res["calibration"].empty
    assert len(res["folds"]) == 5


def test_walk_forward_folds_are_time_ordered(features):
    res = bt.walk_forward_backtest(features, model_kind="logistic", n_splits=4)
    folds = res["folds"]
    # Expanding window: each fold trains on at least as much as the previous.
    assert folds["n_train"].is_monotonic_increasing


def test_tune_returns_best_and_search_table(features):
    grid = {"max_depth": [3], "n_estimators": [50, 150]}
    res = bt.tune_model("xgboost", features=features, param_grid=grid,
                        n_splits=3, verbose=False)
    assert set(res["best_params"]) == {"max_depth", "n_estimators"}
    # Native types, sorted best-first by log loss.
    assert isinstance(res["best_params"]["n_estimators"], int)
    assert res["results"]["log_loss"].is_monotonic_increasing
    assert res["best_score"] == pytest.approx(res["results"]["log_loss"].min())


def test_best_params_roundtrip(tmp_path, monkeypatch):
    from src import config
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    params = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.03}
    common.save_best_params("xgboost", params)
    assert common.load_best_params("xgboost") == params
    # Missing file returns an empty dict, not an error.
    assert common.load_best_params("does_not_exist") == {}

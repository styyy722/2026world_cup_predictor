"""Tests for walk-forward backtesting and hyperparameter tuning."""
import pytest

from src.evaluation import backtest as bt
from src.features import build_features as bf
from src.models import baseline_logistic as logit
from src.models import common


@pytest.fixture(scope="module")
def features():
    return bf.build_training_features()


def test_walk_forward_uses_training_window(features):
    res = bt.walk_forward_backtest(features, model_kind="logistic", n_splits=5)
    # With 5 expanding folds the test set covers most of the filtered matches.
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


def test_selection_roundtrip(tmp_path, monkeypatch):
    from src import config
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    sel = {"model": "lightgbm", "params": {"max_depth": 3},
           "walk_forward_log_loss": 0.98}
    common.save_selection(sel)
    assert common.load_selection() == sel
    # No selection saved yet -> empty dict.
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "empty")
    assert common.load_selection() == {}


def test_calibrated_classifier_predicts_probabilities(features):
    base = logit.train_model(features)
    model = common.calibrate_classifier(
        base,
        features,
        method="sigmoid",
        cv=2,
    )
    row = features.iloc[0].to_dict()
    probs = common.predict_proba_dicts(model, row)
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_tune_and_select_picks_lowest_log_loss(features, monkeypatch):
    # Tiny grids across two fast backends keep the test quick.
    tiny = {
        "xgboost": {"max_depth": [3], "n_estimators": [80]},
        "lightgbm": {"max_depth": [3], "n_estimators": [80]},
    }
    monkeypatch.setattr(bt, "_PARAM_GRIDS", tiny)
    res = bt.tune_and_select(model_kinds=("xgboost", "lightgbm"),
                             features=features, n_splits=3, verbose=False)
    assert res["best_model"] in ("xgboost", "lightgbm")
    # The comparison is sorted best-first and the winner matches it.
    assert res["comparison"].iloc[0]["model"] == res["best_model"]
    assert res["best_score"] == pytest.approx(
        res["comparison"]["best_log_loss"].min())

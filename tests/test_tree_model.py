"""Tests for the tree-based gradient-boosting match models."""
import pytest

from src.features import build_features as bf
from src.models import tree_model as tree


@pytest.fixture(scope="module")
def training_features():
    # A modest synthetic feature table is enough to fit a small booster.
    return bf.build_training_features()


def _installed_backends():
    return tree.available_backends()


def test_at_least_one_backend_available():
    # The project ships with xgboost + lightgbm in requirements.
    assert len(_installed_backends()) >= 1


@pytest.mark.parametrize("backend", _installed_backends())
def test_train_and_predict_probability_sums_to_one(backend, training_features):
    model = tree.train_model(training_features, backend=backend)
    row = training_features.iloc[0].to_dict()
    probs = tree.predict_match_proba(model, row)
    total = probs["team_a_win"] + probs["draw"] + probs["team_b_win"]
    assert abs(total - 1.0) < 1e-6
    assert all(0.0 <= p <= 1.0 for p in probs.values())


@pytest.mark.parametrize("backend", _installed_backends())
def test_batch_prediction_returns_list(backend, training_features):
    model = tree.train_model(training_features, backend=backend)
    batch = training_features.head(5)
    out = tree.predict_match_proba(model, batch)
    assert isinstance(out, list) and len(out) == 5
    for p in out:
        assert abs(sum(p.values()) - 1.0) < 1e-6


@pytest.mark.parametrize("backend", _installed_backends())
def test_save_and_load_roundtrip(backend, training_features, tmp_path):
    model = tree.train_model(training_features.head(300), backend=backend)
    path = tmp_path / f"{backend}.pkl"
    tree.save_model(model, backend=backend, path=path)
    loaded = tree.load_model(backend=backend, path=path)

    row = training_features.iloc[0].to_dict()
    p1 = tree.predict_match_proba(model, row)
    p2 = tree.predict_match_proba(loaded, row)
    assert pytest.approx(p1["team_a_win"], abs=1e-9) == p2["team_a_win"]


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        tree._build_estimator("not_a_real_model")

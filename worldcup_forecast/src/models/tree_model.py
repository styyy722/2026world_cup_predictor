"""Tree-based gradient-boosting match models (primary models).

Supports three interchangeable backends behind one interface:

    * ``xgboost``  -> xgboost.XGBClassifier
    * ``lightgbm`` -> lightgbm.LGBMClassifier
    * ``catboost`` -> catboost.CatBoostClassifier

Each predicts the 3-class win/draw/loss outcome (0 loss / 1 draw / 2 win, from
team_a's perspective) from the engineered match features. Trees handle the
mixed-scale features without standardisation, so unlike the logistic baseline
there is no ``StandardScaler`` step.

Backends are imported lazily so the project still runs if only one is
installed. The multinomial logistic regression in ``baseline_logistic`` is kept
as a comparison baseline.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config
from ..features import build_features as bf
from . import common

SUPPORTED_BACKENDS = ("xgboost", "lightgbm", "catboost")
DEFAULT_BACKEND = "xgboost"


def model_path(backend: str) -> Path:
    """Per-backend on-disk location for a saved model."""
    return config.MODELS_DIR / f"{backend}_model.pkl"


def _build_estimator(backend: str, n_classes: int = 3, **overrides):
    """Instantiate the requested booster with sensible defaults.

    Defaults are modest (shallow trees, moderate estimator counts, mild
    regularisation) to avoid overfitting the relatively small international
    match history. Pass keyword overrides to tune.
    """
    backend = backend.lower()
    if backend == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:  # pragma: no cover - env dependent
            raise ImportError("xgboost is not installed. `pip install xgboost`.") from e
        # The sklearn wrapper infers the multiclass objective and class count
        # from y; setting num_class/objective explicitly here conflicts with
        # that and yields malformed probabilities, so we leave them out.
        params = dict(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, min_child_weight=3,
            eval_metric="mlogloss", n_jobs=-1, random_state=42,
            verbosity=0,
        )
        params.update(overrides)
        return XGBClassifier(**params)

    if backend == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as e:  # pragma: no cover - env dependent
            raise ImportError("lightgbm is not installed. `pip install lightgbm`.") from e
        # LGBMClassifier likewise infers the multiclass objective from y.
        params = dict(
            n_estimators=400, learning_rate=0.05, max_depth=4, num_leaves=15,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            min_child_samples=20,
            n_jobs=-1, random_state=42, verbose=-1,
        )
        params.update(overrides)
        return LGBMClassifier(**params)

    if backend == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as e:  # pragma: no cover - env dependent
            raise ImportError("catboost is not installed. `pip install catboost`.") from e
        params = dict(
            iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=3.0,
            loss_function="MultiClass", random_seed=42, verbose=False,
        )
        params.update(overrides)
        return CatBoostClassifier(**params)

    raise ValueError(
        f"Unknown backend '{backend}'. Choose one of {SUPPORTED_BACKENDS}."
    )


def available_backends() -> list[str]:
    """Return the subset of backends that can actually be imported."""
    found = []
    for b in SUPPORTED_BACKENDS:
        try:
            _build_estimator(b)
            found.append(b)
        except ImportError:
            continue
    return found


def train_model(features: pd.DataFrame | None = None,
                backend: str = DEFAULT_BACKEND, **overrides):
    """Train a gradient-boosting classifier on the feature table.

    ``features`` must contain ``bf.NUMERIC_FEATURES`` plus a ``result`` target.
    If ``None``, features are built from the full historical results.
    """
    if features is None:
        features = bf.build_training_features()
    X = bf.features_to_matrix(features)
    y = features["result"].astype(int)

    model = _build_estimator(backend, n_classes=int(y.nunique()), **overrides)
    model.fit(X, y)
    return model


def predict_match_proba(model, feature_row):
    """Predict labelled WDL probabilities (single dict or list of dicts)."""
    return common.predict_proba_dicts(model, feature_row)


def save_model(model, backend: str = DEFAULT_BACKEND,
               path: str | Path | None = None) -> Path:
    """Persist a trained tree model to disk."""
    return common.save_pickle(model, path or model_path(backend))


def load_model(backend: str = DEFAULT_BACKEND, path: str | Path | None = None):
    """Load a previously saved tree model for ``backend``."""
    return common.load_pickle(
        path or model_path(backend),
        hint=f"Run `python main.py --mode train --model {backend}` first.",
    )

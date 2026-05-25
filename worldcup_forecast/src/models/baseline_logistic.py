"""Multinomial logistic regression match model (baseline).

This is the **baseline** model; the primary models are the gradient-boosted
trees in ``tree_model``. It predicts the three-way outcome
(team_a loss / draw / team_a win) from the engineered match features, encoded:

    0 = team_a loss, 1 = draw, 2 = team_a win

Probability extraction and persistence are shared via ``models.common`` so
every model exposes the same interface to the simulator.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .. import config
from ..features import build_features as bf
from . import common

DEFAULT_MODEL_PATH = config.MODELS_DIR / "logistic_model.pkl"

# The classes the model is trained to predict, in scikit-learn order.
CLASSES = common.CLASSES  # 0 loss, 1 draw, 2 win (team_a perspective)


def train_model(features: pd.DataFrame | None = None,
                C: float = 1.0, max_iter: int = 1000) -> Pipeline:
    """Train a multinomial logistic regression on the feature table.

    ``features`` must contain the columns in ``bf.NUMERIC_FEATURES`` plus a
    ``result`` target column. If ``None``, features are built from the full
    historical results.
    """
    if features is None:
        features = bf.build_training_features()

    X = bf.features_to_matrix(features)
    y = features["result"].astype(int)

    model = Pipeline([
        ("scaler", StandardScaler()),
        # solver="lbfgs" performs multinomial (softmax) regression for the
        # 3-class WDL target in current scikit-learn.
        ("clf", LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")),
    ])
    model.fit(X, y)
    return model


def predict_match_proba(model: Pipeline, feature_row: dict | pd.DataFrame):
    """Predict WDL probabilities for one or more matches.

    With a single dict input returns one labelled dict; with a DataFrame
    returns a list of dicts (one per row).
    """
    return common.predict_proba_dicts(model, feature_row)


def save_model(model: Pipeline, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """Persist a trained model to disk via pickle."""
    return common.save_pickle(model, path)


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a previously saved model."""
    return common.load_pickle(
        path, hint="Run `python main.py --mode train --model logistic` first."
    )

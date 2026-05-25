"""Multinomial logistic regression match model.

Predicts the three-way outcome (team_a loss / draw / team_a win) from the
engineered match features. The result classes are encoded:

    0 = team_a loss, 1 = draw, 2 = team_a win

so the predicted probability vector aligns as
(P_team_b_win, P_draw, P_team_a_win) -> we re-order to a clean dict on output.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .. import config
from ..features import build_features as bf

DEFAULT_MODEL_PATH = config.MODELS_DIR / "logistic_model.pkl"

# The classes the model is trained to predict, in scikit-learn order.
CLASSES = [0, 1, 2]  # 0 loss, 1 draw, 2 win (team_a perspective)


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


def _proba_vector_to_dict(proba_row: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    """Map a model probability row to a labelled WDL dict.

    Robust to scikit-learn dropping a class that never appears in training.
    """
    by_class = {int(c): float(p) for c, p in zip(classes, proba_row)}
    return {
        "team_b_win": by_class.get(0, 0.0),  # class 0 = team_a loss
        "draw": by_class.get(1, 0.0),        # class 1 = draw
        "team_a_win": by_class.get(2, 0.0),  # class 2 = team_a win
    }


def predict_match_proba(model: Pipeline, feature_row: dict | pd.DataFrame) -> dict[str, float]:
    """Predict WDL probabilities for one or more matches.

    With a single dict input returns one labelled dict; with a DataFrame
    returns a list of dicts (one per row).
    """
    single = isinstance(feature_row, dict)
    X = bf.features_to_matrix([feature_row] if single else feature_row)
    proba = model.predict_proba(X)
    classes = model.named_steps["clf"].classes_
    dicts = [_proba_vector_to_dict(row, classes) for row in proba]
    return dicts[0] if single else dicts


def save_model(model: Pipeline, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """Persist a trained model to disk via pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    return path


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a previously saved model."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python main.py --mode train` first."
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)

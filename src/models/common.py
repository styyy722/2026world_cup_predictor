"""Shared helpers for sklearn-compatible match classifiers.

All match models (logistic baseline, tree-based boosters) share the same
3-class target and feature matrix, so probability extraction and
persistence are centralised here. This lets the simulator treat every model
identically via ``predict_proba_dicts``.

Result class encoding (team_a perspective):
    0 = team_a loss, 1 = draw, 2 = team_a win
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .. import config
from ..features import build_features as bf

# Canonical class order used everywhere.
CLASSES = [0, 1, 2]


def model_classes(model) -> np.ndarray:
    """Return a fitted model's class labels, whether Pipeline or bare estimator."""
    if isinstance(model, Pipeline):
        return model.named_steps["clf"].classes_
    return model.classes_


def proba_row_to_wdl(proba_row: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    """Map one probability row to a labelled win/draw/loss dict.

    Robust to a class being absent from the training data.
    """
    by_class = {int(c): float(p) for c, p in zip(classes, proba_row)}
    return {
        "team_b_win": by_class.get(0, 0.0),  # class 0 = team_a loss
        "draw": by_class.get(1, 0.0),        # class 1 = draw
        "team_a_win": by_class.get(2, 0.0),  # class 2 = team_a win
    }


def predict_proba_dicts(model, feature_row):
    """Predict labelled WDL probabilities for one match (dict) or many (DataFrame).

    Returns a single dict for a dict input, or a list of dicts for a DataFrame.
    """
    single = isinstance(feature_row, dict)
    X = bf.features_to_matrix([feature_row] if single else feature_row)
    proba = model.predict_proba(X)
    classes = model_classes(model)
    dicts = [proba_row_to_wdl(row, classes) for row in proba]
    return dicts[0] if single else dicts


def save_pickle(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    return path


def load_pickle(path: str | Path, hint: str = ""):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. {hint}".strip()
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _params_path(name: str) -> Path:
    return config.MODELS_DIR / f"{name}_best_params.json"


def save_best_params(name: str, params: dict) -> Path:
    """Persist tuned hyperparameters for a model so training can reuse them."""
    path = _params_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(params, fh, indent=2)
    return path


def load_best_params(name: str) -> dict:
    """Load tuned hyperparameters for a model, or ``{}`` if none saved."""
    path = _params_path(name)
    if not path.exists():
        return {}
    with open(path) as fh:
        return json.load(fh)


def _selection_path() -> Path:
    return config.MODELS_DIR / "selected_model.json"


def save_selection(selection: dict) -> Path:
    """Persist which model was selected as best (and its tuned params)."""
    path = _selection_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(selection, fh, indent=2)
    return path


def load_selection() -> dict:
    """Load the selected-best-model record, or ``{}`` if none saved."""
    path = _selection_path()
    if not path.exists():
        return {}
    with open(path) as fh:
        return json.load(fh)

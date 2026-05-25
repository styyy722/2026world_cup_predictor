"""Backtesting the match model on past World Cups.

Strategy: for each target World Cup year, train only on matches strictly
before that tournament and evaluate on the World Cup matches of that year.
This time-respecting split avoids look-ahead leakage.

Metrics: accuracy, multiclass log loss, multiclass Brier score, and a
calibration table that buckets predicted probabilities against realised
frequencies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from ..features import build_features as bf
from ..models import baseline_logistic as logit
from ..models import common
from ..models import tree_model as tree

# Class order used everywhere: 0 loss, 1 draw, 2 win (team_a perspective).
_CLASSES = [0, 1, 2]


def _train_for_kind(model_kind: str, train_df: pd.DataFrame):
    """Train a model of the requested kind on the training slice."""
    if model_kind == "logistic":
        return logit.train_model(train_df)
    if model_kind in tree.SUPPORTED_BACKENDS:
        return tree.train_model(train_df, backend=model_kind)
    raise ValueError(f"Unsupported backtest model '{model_kind}'.")


def _is_world_cup_match(tournament: str) -> bool:
    t = str(tournament).lower()
    return ("world cup" in t) and ("qualif" not in t)


def _proba_matrix(model, X: pd.DataFrame) -> np.ndarray:
    """Return an (n, 3) probability matrix aligned to classes [0,1,2]."""
    proba = model.predict_proba(X)
    classes = list(common.model_classes(model))
    out = np.zeros((len(X), 3))
    for j, c in enumerate(classes):
        out[:, _CLASSES.index(int(c))] = proba[:, j]
    # If a class was absent from the training slice its column is all zeros;
    # renormalise so each row is a valid probability distribution over [0,1,2].
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return out / row_sums


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean squared error between one-hot truth and predicted probabilities."""
    onehot = np.zeros_like(proba)
    for i, y in enumerate(y_true):
        onehot[i, _CLASSES.index(int(y))] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def calibration_table(y_true: np.ndarray, proba: np.ndarray,
                      n_bins: int = 10) -> pd.DataFrame:
    """Calibration of the predicted team_a-win probability.

    Buckets the P(team_a_win) predictions and compares mean predicted prob to
    the observed team_a-win frequency in each bucket.
    """
    p_win = proba[:, _CLASSES.index(2)]
    actual_win = (np.asarray(y_true) == 2).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p_win, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append({
            "bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
            "n": int(mask.sum()),
            "mean_predicted": float(p_win[mask].mean()),
            "observed_freq": float(actual_win[mask].mean()),
        })
    return pd.DataFrame(rows)


def backtest_world_cup(features: pd.DataFrame, results: pd.DataFrame,
                       year: int, model_kind: str = "logistic") -> dict | None:
    """Train on pre-``year`` data, test on that year's World Cup matches.

    ``features`` is the full training feature table (aligned row-for-row with
    ``results``, which carries ``date`` and ``tournament``). ``model_kind``
    selects the model ("logistic" or any tree backend). Returns a metrics dict,
    or ``None`` if there are no World Cup matches for that year.
    """
    feats = features.reset_index(drop=True).copy()
    meta = results.reset_index(drop=True)
    feats["date"] = meta["date"].values
    feats["tournament"] = meta["tournament"].values

    cutoff = pd.Timestamp(f"{year}-05-01")
    is_wc = feats["tournament"].apply(_is_world_cup_match)
    is_year = feats["date"].dt.year == year

    train = feats[feats["date"] < cutoff]
    test = feats[is_wc & is_year]
    if test.empty or train.empty:
        return None

    model = _train_for_kind(model_kind, train)
    X_test = bf.features_to_matrix(test)
    y_test = test["result"].astype(int).values

    proba = _proba_matrix(model, X_test)
    preds = proba.argmax(axis=1)  # index in [0,1,2] == class label

    return {
        "model": model_kind,
        "year": year,
        "n_matches": int(len(test)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "log_loss": float(log_loss(y_test, proba, labels=_CLASSES)),
        "brier": multiclass_brier(y_test, proba),
        "calibration": calibration_table(y_test, proba),
    }


def run_backtests(years: tuple[int, ...] = (2014, 2018, 2022),
                  model_kinds: tuple[str, ...] = ("logistic", "xgboost")
                  ) -> pd.DataFrame:
    """Backtest each model kind across the target World Cups and summarise."""
    from ..data import loaders
    results = loaders.load_results()
    features = bf.build_training_features(results)

    rows = []
    for model_kind in model_kinds:
        for y in years:
            res = backtest_world_cup(features, results, y, model_kind=model_kind)
            if res is None:
                print(f"[backtest] No World Cup matches for {y}; skipping.")
                continue
            rows.append({k: res[k] for k in ("model", "year", "n_matches",
                                             "accuracy", "log_loss", "brier")})
            print(f"[backtest] {model_kind:>8} {y}: n={res['n_matches']} "
                  f"acc={res['accuracy']:.3f} logloss={res['log_loss']:.3f} "
                  f"brier={res['brier']:.3f}")
    return pd.DataFrame(rows)

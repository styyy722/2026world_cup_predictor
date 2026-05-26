"""Backtesting and hyperparameter tuning for the match models.

Two complementary backtests:

* :func:`backtest_world_cup` - train on everything before a target World Cup,
  test on that tournament's matches (2014/2018/2022). A realistic but small
  test set.
* :func:`walk_forward_backtest` - expanding-window time-series cross-validation
  over the configured training window, so recent periods after the first fold
  are used as out-of-sample test data. This is the metric we tune against.

:func:`tune_model` searches a hyperparameter grid using the walk-forward log
loss (a proper scoring rule) as the objective and returns the best settings,
which are persisted so training picks them up automatically.

Metrics: accuracy, multiclass log loss, multiclass Brier score, and a
calibration table.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..features import build_features as bf
from ..models import baseline_logistic as logit
from ..models import common
from ..models import tree_model as tree

# Class order used everywhere: 0 loss, 1 draw, 2 win (team_a perspective).
_CLASSES = [0, 1, 2]


def _train_for_kind(model_kind: str, train_df: pd.DataFrame, params: dict | None = None):
    """Train a model of the requested kind on a feature DataFrame slice."""
    params = params or {}
    if model_kind == "logistic":
        return logit.train_model(train_df, **params)
    if model_kind in tree.SUPPORTED_BACKENDS:
        return tree.train_model(train_df, backend=model_kind, **params)
    raise ValueError(f"Unsupported backtest model '{model_kind}'.")


def _build_and_fit(model_kind: str, X: pd.DataFrame, y: np.ndarray,
                   params: dict | None = None):
    """Build and fit an estimator directly from a feature matrix and labels."""
    params = params or {}
    if model_kind == "logistic":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="lbfgs", max_iter=1000,
                                       **params)),
        ])
        model.fit(X, y)
        return model
    if model_kind in tree.SUPPORTED_BACKENDS:
        est = tree._build_estimator(model_kind, n_classes=len(set(y)), **params)
        est.fit(X, y)
        return est
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
                       year: int, model_kind: str = "logistic",
                       params: dict | None = None) -> dict | None:
    """Train on pre-``year`` data, test on that year's World Cup matches.

    ``features`` is the full training feature table (aligned row-for-row with
    ``results``, which carries ``date`` and ``tournament``). ``model_kind``
    selects the model ("logistic" or any tree backend); ``params`` overrides
    its hyperparameters. Returns a metrics dict, or ``None`` if there are no
    World Cup matches for that year.
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

    model = _train_for_kind(model_kind, train, params)
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


def walk_forward_backtest(features: pd.DataFrame, model_kind: str = "xgboost",
                          params: dict | None = None, n_splits: int = 6
                          ) -> dict:
    """Expanding-window time-series CV over the whole match history.

    ``features`` must be in chronological order (``build_training_features``
    returns it sorted by date). Each fold trains on all earlier matches and
    tests on the next block, so collectively every match after the first fold
    is scored out-of-sample. Predictions are pooled across folds for overall
    accuracy / log loss / Brier and a single calibration table.
    """
    X = bf.features_to_matrix(features).reset_index(drop=True)
    y = features["result"].astype(int).to_numpy()

    tscv = TimeSeriesSplit(n_splits=n_splits)
    pooled_y, pooled_p = [], []
    fold_rows = []
    for fold, (tr, te) in enumerate(tscv.split(X), start=1):
        model = _build_and_fit(model_kind, X.iloc[tr], y[tr], params)
        proba = _proba_matrix(model, X.iloc[te])
        yt = y[te]
        pooled_y.append(yt)
        pooled_p.append(proba)
        fold_rows.append({
            "fold": fold,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "accuracy": float(accuracy_score(yt, proba.argmax(axis=1))),
            "log_loss": float(log_loss(yt, proba, labels=_CLASSES)),
        })

    y_all = np.concatenate(pooled_y)
    p_all = np.vstack(pooled_p)
    return {
        "model": model_kind,
        "n_splits": n_splits,
        "n_test_total": int(len(y_all)),
        "accuracy": float(accuracy_score(y_all, p_all.argmax(axis=1))),
        "log_loss": float(log_loss(y_all, p_all, labels=_CLASSES)),
        "brier": multiclass_brier(y_all, p_all),
        "calibration": calibration_table(y_all, p_all),
        "folds": pd.DataFrame(fold_rows),
    }


# Default tuning grids per model. Kept modest so a full search runs in minutes.
_PARAM_GRIDS = {
    "xgboost": {
        "n_estimators": [200, 400, 600],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.03, 0.05, 0.1],
    },
    "lightgbm": {
        "n_estimators": [200, 400, 600],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.03, 0.05, 0.1],
    },
    "catboost": {
        "iterations": [200, 400, 600],
        "depth": [3, 4, 6],
        "learning_rate": [0.03, 0.05, 0.1],
    },
    "logistic": {
        "C": [0.1, 0.3, 1.0, 3.0, 10.0],
    },
}


def _grid_combinations(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]


def tune_model(model_kind: str = "xgboost", features: pd.DataFrame | None = None,
               param_grid: dict | None = None, n_splits: int = 5,
               verbose: bool = True) -> dict:
    """Grid-search hyperparameters using walk-forward log loss as the objective.

    Returns a dict with ``best_params``, ``best_score`` (log loss), the baseline
    (default-params) score for comparison, and a ``results`` DataFrame of every
    configuration tried, sorted best-first.
    """
    if features is None:
        features = bf.build_training_features()
    grid = param_grid or _PARAM_GRIDS.get(model_kind)
    if grid is None:
        raise ValueError(f"No tuning grid for model '{model_kind}'.")

    combos = _grid_combinations(grid)
    baseline = walk_forward_backtest(features, model_kind, params=None,
                                     n_splits=n_splits)
    if verbose:
        print(f"[tune] {model_kind}: searching {len(combos)} configs "
              f"(baseline logloss={baseline['log_loss']:.4f}, "
              f"acc={baseline['accuracy']:.4f})")

    rows = []
    for i, params in enumerate(combos, start=1):
        res = walk_forward_backtest(features, model_kind, params=params,
                                    n_splits=n_splits)
        rows.append({**params, "log_loss": res["log_loss"],
                     "accuracy": res["accuracy"], "brier": res["brier"]})
        if verbose:
            print(f"[tune]  ({i}/{len(combos)}) {params} -> "
                  f"logloss={res['log_loss']:.4f} acc={res['accuracy']:.4f}")

    results = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    best = results.iloc[0]
    # Cast to native Python types (int where the grid used ints) for clean JSON.
    best_params = {}
    for k in grid.keys():
        v = best[k]
        if all(isinstance(g, int) for g in grid[k]):
            best_params[k] = int(v)
        else:
            best_params[k] = float(v)

    if verbose:
        print(f"[tune] best {model_kind}: {best_params} "
              f"logloss={best['log_loss']:.4f} acc={best['accuracy']:.4f} "
              f"(baseline {baseline['log_loss']:.4f}/{baseline['accuracy']:.4f})")

    return {
        "model": model_kind,
        "best_params": best_params,
        "best_score": float(best["log_loss"]),
        "best_accuracy": float(best["accuracy"]),
        "baseline_score": float(baseline["log_loss"]),
        "baseline_accuracy": float(baseline["accuracy"]),
        "results": results,
    }


# The three gradient-boosting backends we compare by default.
BOOSTING_MODELS = ("xgboost", "lightgbm", "catboost")


def tune_and_select(model_kinds: tuple[str, ...] = BOOSTING_MODELS,
                    features: pd.DataFrame | None = None, n_splits: int = 4,
                    verbose: bool = True) -> dict:
    """Tune every available model and select the best by walk-forward log loss.

    Rather than assume one booster is best, this tunes each installed backend
    and picks the winner empirically (lowest walk-forward log loss, a proper
    scoring rule; accuracy is reported as a secondary view).

    Returns a dict with the winning ``best_model`` / ``best_params`` /
    ``best_score``, a ``per_model`` map of each backend's tuning result, and a
    ``comparison`` DataFrame summarising all backends (best-first).
    """
    if features is None:
        features = bf.build_training_features()

    available = [m for m in model_kinds if m in tree.available_backends()]
    missing = [m for m in model_kinds if m not in available]
    if missing:
        print(f"[select] skipping unavailable backends (not installed): {missing}")
    if not available:
        raise RuntimeError("No requested boosting backends are installed.")

    per_model: dict[str, dict] = {}
    rows = []
    for m in available:
        if verbose:
            print(f"[select] === tuning {m} ===")
        res = tune_model(m, features=features, n_splits=n_splits, verbose=verbose)
        per_model[m] = res
        rows.append({
            "model": m,
            "best_log_loss": res["best_score"],
            "best_accuracy": res["best_accuracy"],
            "default_log_loss": res["baseline_score"],
            "default_accuracy": res["baseline_accuracy"],
            "best_params": res["best_params"],
        })

    comparison = pd.DataFrame(rows).sort_values("best_log_loss").reset_index(drop=True)
    best_model = comparison.iloc[0]["model"]
    if verbose:
        print("\n[select] model comparison (walk-forward, tuned):")
        for _, r in comparison.iterrows():
            print(f"[select]  {r['model']:>9}: logloss={r['best_log_loss']:.4f} "
                  f"acc={r['best_accuracy']:.4f}  params={r['best_params']}")
        print(f"[select] WINNER: {best_model}")

    return {
        "best_model": best_model,
        "best_params": per_model[best_model]["best_params"],
        "best_score": per_model[best_model]["best_score"],
        "best_accuracy": per_model[best_model]["best_accuracy"],
        "per_model": per_model,
        "comparison": comparison,
    }


def run_backtests(years: tuple[int, ...] = (2006, 2010, 2014, 2018, 2022),
                  model_kinds: tuple[str, ...] = ("logistic", "xgboost"),
                  walk_forward: bool = True, n_splits: int = 6
                  ) -> pd.DataFrame:
    """Backtest each model kind and summarise.

    Runs the per-World-Cup backtests and, when ``walk_forward`` is set, the
    expanding-window CV over the configured training window. Returns a combined
    summary DataFrame (``scope`` column distinguishes ``wc:<year>`` rows from
    ``walk_forward``).
    """
    from ..data import loaders
    raw_results = loaders.load_results()
    results = bf.filter_training_results(raw_results)
    features = bf.build_training_features(results, apply_training_window=False)

    rows = []
    for model_kind in model_kinds:
        # Use tuned hyperparameters if a tuning run has saved them, so the
        # backtest reflects the model that training actually deploys.
        params = common.load_best_params(model_kind) or None
        if params:
            print(f"[backtest] {model_kind}: using tuned params {params}")

        for y in years:
            res = backtest_world_cup(features, results, y, model_kind=model_kind,
                                     params=params)
            if res is None:
                print(f"[backtest] No World Cup matches for {y}; skipping.")
                continue
            rows.append({
                "model": model_kind, "scope": f"wc:{y}",
                "n_matches": res["n_matches"], "accuracy": res["accuracy"],
                "log_loss": res["log_loss"], "brier": res["brier"],
            })
            print(f"[backtest] {model_kind:>8} wc:{y}: n={res['n_matches']} "
                  f"acc={res['accuracy']:.3f} logloss={res['log_loss']:.3f} "
                  f"brier={res['brier']:.3f}")

        if walk_forward:
            wf = walk_forward_backtest(features, model_kind, params=params,
                                       n_splits=n_splits)
            rows.append({
                "model": model_kind, "scope": "walk_forward",
                "n_matches": wf["n_test_total"], "accuracy": wf["accuracy"],
                "log_loss": wf["log_loss"], "brier": wf["brier"],
            })
            print(f"[backtest] {model_kind:>8} walk_forward: "
                  f"n={wf['n_test_total']} acc={wf['accuracy']:.3f} "
                  f"logloss={wf['log_loss']:.3f} brier={wf['brier']:.3f}")
    return pd.DataFrame(rows)

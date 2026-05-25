"""Command-line entry point for the 2026 World Cup forecasting pipeline.

Usage:
    python main.py --mode train   [--model xgboost|lightgbm|catboost|logistic]
    python main.py --mode tune    [--model xgboost|lightgbm|logistic]
    python main.py --mode simulate --n_simulations 10000
    python main.py --mode full --n_simulations 10000 [--model ...]
    python main.py --mode backtest [--model ...]

``full`` runs the whole pipeline: load data -> build features -> train ->
predict fixtures -> simulate tournament -> export tables -> export charts.
``tune`` searches hyperparameters via walk-forward CV and saves the best ones,
which ``train``/``full`` then pick up automatically.
"""
from __future__ import annotations

import argparse

import pandas as pd

from src import config
from src.data import loaders
from src.evaluation import backtest as bt
from src.features import build_features as bf
from src.models import baseline_logistic as logit
from src.models import common
from src.models import tree_model as tree
from src.simulation import simulate_tournament as sim
from src.simulation.predictor import MatchPredictor
from src.visualisation import charts

# Default primary model: gradient-boosted trees. The logistic regression is
# kept as a baseline and is also trained for comparison.
DEFAULT_MODEL = "xgboost"
_TREE_BACKENDS = ("xgboost", "lightgbm", "catboost")


def _build_predictor(model_kind: str) -> MatchPredictor:
    """Construct the requested predictor with latest ratings and form."""
    ratings = loaders.latest_ratings()
    if model_kind == "elo":
        return MatchPredictor.from_elo(ratings)

    form = bf.current_form_table()
    if model_kind == "logistic":
        model = logit.load_model()
    elif model_kind in _TREE_BACKENDS:
        model = tree.load_model(model_kind)
    else:
        raise ValueError(f"Unknown model '{model_kind}'.")
    return MatchPredictor.from_classifier(model_kind, model, ratings, form)


def run_train(model_kind: str = DEFAULT_MODEL) -> None:
    """Train the selected primary model plus the logistic baseline, and save.

    The logistic regression is always trained as a baseline for comparison; the
    selected tree backend (default XGBoost) is the primary forecasting model.
    """
    print("[train] Building training features from historical results...")
    features = bf.build_training_features()
    print(f"[train] {len(features)} historical matches in training set.")

    # Baseline (applies tuned params if a tuning run saved them).
    base_params = common.load_best_params("logistic")
    base = logit.train_model(features, **base_params)
    base_path = logit.save_model(base)
    tuned_note = " (tuned)" if base_params else ""
    print(f"[train] Saved logistic baseline{tuned_note} to {base_path}")

    # Primary tree model (skip duplicate work if user picked logistic/elo).
    if model_kind in _TREE_BACKENDS:
        params = common.load_best_params(model_kind)
        tmodel = tree.train_model(features, backend=model_kind, **params)
        tpath = tree.save_model(tmodel, backend=model_kind)
        tuned_note = " (tuned params)" if params else " (default params)"
        print(f"[train] Saved primary {model_kind} model{tuned_note} to {tpath}")
    elif model_kind not in ("logistic", "elo"):
        raise ValueError(f"Unknown model '{model_kind}'.")


def run_simulate(n_simulations: int, model_kind: str) -> None:
    """Simulate the tournament and export all tables/charts."""
    config.ensure_dirs()
    groups = loaders.load_groups()
    fixtures = loaders.load_fixtures()
    predictor = _build_predictor(model_kind)

    print(f"[simulate] Predicting {len(fixtures)} group fixtures...")
    match_probs = sim.fixture_match_probabilities(predictor, fixtures)
    match_probs.to_csv(config.TABLES_DIR / "match_probabilities.csv", index=False)

    print(f"[simulate] Running {n_simulations:,} tournament simulations "
          f"({model_kind} model)...")
    res = sim.run_simulations(predictor, groups, fixtures, n_simulations)

    res["team_stage"].to_csv(
        config.TABLES_DIR / "team_stage_probabilities.csv", index=False)
    res["summary"].to_csv(
        config.TABLES_DIR / "simulation_summary.csv", index=False)
    print(f"[simulate] Wrote tables to {config.TABLES_DIR}")

    paths = charts.generate_all_charts(res["team_stage"])
    print(f"[simulate] Wrote {len(paths)} charts to {config.CHARTS_DIR}")

    _print_top(res["team_stage"])


def run_backtest(model_kind: str = DEFAULT_MODEL) -> None:
    """Backtest the primary model against the logistic baseline.

    Runs both per-World-Cup backtests and the expanding-window walk-forward
    cross-validation over the entire match history.
    """
    models = ["logistic"]
    if model_kind in _TREE_BACKENDS and model_kind not in models:
        models.append(model_kind)
    print(f"[backtest] Comparing {models}: World Cups + walk-forward CV...")
    summary = bt.run_backtests(model_kinds=tuple(models), walk_forward=True)
    if not summary.empty:
        out = config.TABLES_DIR / "backtest_summary.csv"
        config.ensure_dirs()
        summary.to_csv(out, index=False)
        print(f"[backtest] Wrote {out}")


def run_tune(model_kind: str = DEFAULT_MODEL) -> None:
    """Tune hyperparameters via walk-forward CV, save them, and retrain."""
    config.ensure_dirs()
    if model_kind == "elo":
        print("[tune] Elo has no trainable parameters; nothing to tune.")
        return

    features = bf.build_training_features()
    res = bt.tune_model(model_kind, features=features)

    params_path = common.save_best_params(model_kind, res["best_params"])
    print(f"[tune] Saved best params to {params_path}")

    # Save the full search results for inspection.
    out = config.TABLES_DIR / f"tuning_results_{model_kind}.csv"
    res["results"].to_csv(out, index=False)
    print(f"[tune] Wrote search results to {out}")

    # Retrain and persist the tuned model so simulate/full use it immediately.
    if model_kind in _TREE_BACKENDS:
        model = tree.train_model(features, backend=model_kind, **res["best_params"])
        tree.save_model(model, backend=model_kind)
    else:  # logistic
        model = logit.train_model(features, **res["best_params"])
        logit.save_model(model)

    delta = res["baseline_score"] - res["best_score"]
    print(f"[tune] log loss {res['baseline_score']:.4f} -> {res['best_score']:.4f} "
          f"(improvement {delta:+.4f}); accuracy "
          f"{res['baseline_accuracy']:.4f} -> {res['best_accuracy']:.4f}")


def _print_top(team_stage: pd.DataFrame, n: int = 10) -> None:
    print("\nTop title contenders:")
    cols = ["team", "group", "prob_reach_sf", "prob_reach_final", "prob_champion"]
    top = team_stage.nlargest(n, "prob_champion")[cols].copy()
    for c in ("prob_reach_sf", "prob_reach_final", "prob_champion"):
        top[c] = (top[c] * 100).round(1).astype(str) + "%"
    print(top.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="2026 World Cup forecaster")
    parser.add_argument("--mode", required=True,
                        choices=["train", "simulate", "full", "backtest", "tune"])
    parser.add_argument("--n_simulations", type=int, default=config.QUICK_SIMULATIONS)
    parser.add_argument(
        "--model",
        choices=["xgboost", "lightgbm", "catboost", "logistic", "elo"],
        default=DEFAULT_MODEL,
        help="Match model to drive the simulation. Tree backends are the "
             "primary models; 'logistic' is the baseline.",
    )
    args = parser.parse_args()

    if args.mode == "train":
        run_train(args.model)
    elif args.mode == "simulate":
        run_simulate(args.n_simulations, args.model)
    elif args.mode == "backtest":
        run_backtest(args.model)
    elif args.mode == "tune":
        run_tune(args.model)
    elif args.mode == "full":
        run_train(args.model)
        run_simulate(args.n_simulations, args.model)


if __name__ == "__main__":
    main()

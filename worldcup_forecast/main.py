"""Command-line entry point for the 2026 World Cup forecasting pipeline.

Usage:
    python main.py --mode train
    python main.py --mode simulate --n_simulations 10000
    python main.py --mode full --n_simulations 10000 [--model logistic|elo]
    python main.py --mode backtest

``full`` runs the whole pipeline: load data -> build features -> train ->
predict fixtures -> simulate tournament -> export tables -> export charts.
"""
from __future__ import annotations

import argparse

import pandas as pd

from src import config
from src.data import loaders
from src.evaluation import backtest as bt
from src.features import build_features as bf
from src.models import baseline_logistic as logit
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

    # Baseline.
    base = logit.train_model(features)
    base_path = logit.save_model(base)
    print(f"[train] Saved logistic baseline to {base_path}")

    # Primary tree model (skip duplicate work if user picked logistic/elo).
    if model_kind in _TREE_BACKENDS:
        tmodel = tree.train_model(features, backend=model_kind)
        tpath = tree.save_model(tmodel, backend=model_kind)
        print(f"[train] Saved primary {model_kind} model to {tpath}")
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
    """Backtest the primary model against the logistic baseline."""
    models = ["logistic"]
    if model_kind in _TREE_BACKENDS and model_kind not in models:
        models.append(model_kind)
    print(f"[backtest] Comparing {models} on World Cups (2014/2018/2022)...")
    summary = bt.run_backtests(model_kinds=tuple(models))
    if not summary.empty:
        out = config.TABLES_DIR / "backtest_summary.csv"
        config.ensure_dirs()
        summary.to_csv(out, index=False)
        print(f"[backtest] Wrote {out}")


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
                        choices=["train", "simulate", "full", "backtest"])
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
    elif args.mode == "full":
        run_train(args.model)
        run_simulate(args.n_simulations, args.model)


if __name__ == "__main__":
    main()

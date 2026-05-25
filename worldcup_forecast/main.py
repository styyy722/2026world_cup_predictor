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
from src.simulation import simulate_tournament as sim
from src.simulation.predictor import MatchPredictor
from src.visualisation import charts


def _build_predictor(model_kind: str) -> MatchPredictor:
    """Construct the requested predictor with latest ratings and form."""
    ratings = loaders.latest_ratings()
    if model_kind == "elo":
        return MatchPredictor.from_elo(ratings)
    model = logit.load_model()
    form = bf.current_form_table()
    return MatchPredictor.from_logistic(model, ratings, form)


def run_train() -> None:
    """Train the logistic model on all historical results and save it."""
    print("[train] Building training features from historical results...")
    features = bf.build_training_features()
    print(f"[train] {len(features)} historical matches in training set.")
    model = logit.train_model(features)
    path = logit.save_model(model)
    print(f"[train] Saved logistic model to {path}")


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


def run_backtest() -> None:
    print("[backtest] Running World Cup backtests (2014/2018/2022)...")
    summary = bt.run_backtests()
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
    parser.add_argument("--model", choices=["logistic", "elo"], default="logistic",
                        help="Match model to drive the simulation.")
    args = parser.parse_args()

    if args.mode == "train":
        run_train()
    elif args.mode == "simulate":
        run_simulate(args.n_simulations, args.model)
    elif args.mode == "backtest":
        run_backtest()
    elif args.mode == "full":
        run_train()
        run_simulate(args.n_simulations, args.model)


if __name__ == "__main__":
    main()

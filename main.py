"""Command-line entry point for the 2026 World Cup forecasting pipeline.

Usage:
    python main.py --mode validate-data
    python main.py --mode train   [--model xgboost|lightgbm|catboost|logistic]
                                  [--calibration none|sigmoid|isotonic]
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

# Default primary model: gradient-boosted trees. The logistic regression is
# kept as a baseline and is also trained for comparison.
DEFAULT_MODEL = "xgboost"
_TREE_BACKENDS = ("xgboost", "lightgbm", "catboost")


def _training_features() -> pd.DataFrame:
    """Build filtered training features and print the active data scope."""
    results = loaders.load_results()
    scope = bf.training_window_summary(results)
    cutoff = scope["cutoff_date"]
    latest = scope["latest_date"]
    if cutoff is not None and latest is not None:
        print(
            "[train] Scope: "
            f"{scope['kept_matches']}/{scope['total_matches']} matches kept; "
            f"non-World-Cup matches from {cutoff.date()} to {latest.date()}, "
            f"World Cup proper limited to {scope['last_world_cup_year']}."
        )
    return bf.build_training_features(results)


def _resolve_model(model_kind: str) -> str:
    """Resolve 'best' to the model chosen by `--mode select`."""
    if model_kind != "best":
        return model_kind
    sel = common.load_selection()
    if not sel:
        raise ValueError(
            "No selected model found. Run `python main.py --mode select` first."
        )
    print(f"[model] using selected best model: {sel['model']}")
    return sel["model"]


def _build_predictor(model_kind: str, odds_weight: float = 0.0,
                     odds_method: str = "shin",
                     odds_blend: str = "logarithmic") -> MatchPredictor:
    """Construct the requested predictor with latest ratings and form."""
    model_kind = _resolve_model(model_kind)
    ratings = loaders.latest_ratings()
    odds = loaders.load_betting_odds()
    if model_kind == "elo":
        return MatchPredictor.from_elo(
            ratings,
            odds=odds,
            odds_weight=odds_weight,
            odds_method=odds_method,
            odds_blend=odds_blend,
        )

    form = bf.current_form_table()
    context = loaders.load_team_context()
    optional_context = {
        "player_status": loaders.load_player_status(),
        "player_form": loaders.load_player_form(),
        "team_status": loaders.load_team_status(),
        "match_context": loaders.load_match_context(),
    }
    if model_kind == "logistic":
        model = logit.load_model()
    elif model_kind in _TREE_BACKENDS:
        model = tree.load_model(model_kind)
    else:
        raise ValueError(f"Unknown model '{model_kind}'.")
    return MatchPredictor.from_classifier(
        model_kind,
        model,
        ratings,
        form,
        context=context,
        optional_context=optional_context,
        odds=odds,
        odds_weight=odds_weight,
        odds_method=odds_method,
        odds_blend=odds_blend,
    )


def run_train(model_kind: str = DEFAULT_MODEL,
              calibration: str = "none") -> None:
    """Train the selected primary model plus the logistic baseline, and save.

    The logistic regression is always trained as a baseline for comparison; the
    selected tree backend (default XGBoost) is the primary forecasting model.
    """
    model_kind = _resolve_model(model_kind)
    print("[train] Building training features from historical results...")
    features = _training_features()
    print(f"[train] {len(features)} historical matches in training set.")

    # Baseline (applies tuned params if a tuning run saved them).
    base_params = common.load_best_params("logistic")
    base = logit.train_model(features, **base_params)
    if calibration != "none":
        print(f"[train] Calibrating logistic baseline ({calibration})...")
        base = common.calibrate_classifier(base, features, method=calibration)
    base_path = logit.save_model(base)
    tuned_note = " (tuned)" if base_params else ""
    calibrated_note = f" + {calibration} calibration" if calibration != "none" else ""
    print(f"[train] Saved logistic baseline{tuned_note}{calibrated_note} to {base_path}")

    # Primary tree model (skip duplicate work if user picked logistic/elo).
    if model_kind in _TREE_BACKENDS:
        params = common.load_best_params(model_kind)
        tmodel = tree.train_model(features, backend=model_kind, **params)
        if calibration != "none":
            print(f"[train] Calibrating {model_kind} model ({calibration})...")
            tmodel = common.calibrate_classifier(tmodel, features, method=calibration)
        tpath = tree.save_model(tmodel, backend=model_kind)
        tuned_note = " (tuned params)" if params else " (default params)"
        calibrated_note = f" + {calibration} calibration" if calibration != "none" else ""
        print(
            f"[train] Saved primary {model_kind} model"
            f"{tuned_note}{calibrated_note} to {tpath}"
        )
    elif model_kind not in ("logistic", "elo"):
        raise ValueError(f"Unknown model '{model_kind}'.")


def run_simulate(n_simulations: int, model_kind: str,
                 odds_weight: float = 0.0,
                 odds_method: str = "shin",
                 odds_blend: str = "logarithmic",
                 simulation_seed: int = 2026) -> None:
    """Simulate the tournament and export all tables/charts."""
    from src.visualisation import charts

    config.ensure_dirs()
    groups = loaders.load_groups()
    fixtures = loaders.load_fixtures()
    predictor = _build_predictor(
        model_kind,
        odds_weight=odds_weight,
        odds_method=odds_method,
        odds_blend=odds_blend,
    )

    print(f"[simulate] Predicting {len(fixtures)} group fixtures...")
    match_probs = sim.fixture_match_probabilities(predictor, fixtures)
    match_probs.to_csv(config.TABLES_DIR / "match_probabilities.csv", index=False)

    print(f"[simulate] Running {n_simulations:,} tournament simulations "
          f"({model_kind} model, seed={simulation_seed})...")
    res = sim.run_simulations(
        predictor,
        groups,
        fixtures,
        n_simulations,
        seed=simulation_seed,
    )

    res["team_stage"].to_csv(
        config.TABLES_DIR / "team_stage_probabilities.csv", index=False)
    res["summary"].to_csv(
        config.TABLES_DIR / "simulation_summary.csv", index=False)
    print(f"[simulate] Wrote tables to {config.TABLES_DIR}")

    paths = charts.generate_all_charts(res["team_stage"])
    print(f"[simulate] Wrote {len(paths)} charts to {config.CHARTS_DIR}")

    _print_top(res["team_stage"])


def run_backtest(model_kind: str = DEFAULT_MODEL,
                 years: tuple[int, ...] | None = None) -> None:
    """Backtest the primary model against the logistic baseline.

    Runs per-World-Cup backtests (default 2006/2010/2014/2018/2022) and the
    expanding-window walk-forward cross-validation over the entire match
    history. ``years`` overrides which World Cups are tested.
    """
    if model_kind == "best":
        model_kind = _resolve_model(model_kind)
    models = ["logistic"]
    if model_kind in _TREE_BACKENDS and model_kind not in models:
        models.append(model_kind)
    kwargs = {"model_kinds": tuple(models), "walk_forward": True}
    if years:
        kwargs["years"] = tuple(years)
    print(f"[backtest] Comparing {models}: World Cups + walk-forward CV...")
    summary = bt.run_backtests(**kwargs)
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

    features = _training_features()
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


def run_select() -> None:
    """Tune all three boosters, compare, and select + persist the best.

    Tunes XGBoost, LightGBM and CatBoost (whichever are installed) via
    walk-forward CV, picks the one with the lowest log loss, saves each
    backend's tuned params and the overall selection, writes a comparison
    table, and retrains the winner so simulate/full can use `--model best`.
    """
    config.ensure_dirs()
    features = _training_features()
    res = bt.tune_and_select(features=features)

    # Persist every backend's tuned params (so any can be used later).
    for model_kind, mres in res["per_model"].items():
        common.save_best_params(model_kind, mres["best_params"])
        res_path = config.TABLES_DIR / f"tuning_results_{model_kind}.csv"
        mres["results"].to_csv(res_path, index=False)

    # Persist the comparison and the selection record.
    res["comparison"].to_csv(config.TABLES_DIR / "model_selection.csv", index=False)
    selection = {
        "model": res["best_model"],
        "params": res["best_params"],
        "walk_forward_log_loss": res["best_score"],
        "walk_forward_accuracy": res["best_accuracy"],
    }
    sel_path = common.save_selection(selection)
    print(f"[select] Saved selection ({res['best_model']}) to {sel_path}")

    # Retrain and save the winning model so `--model best` works immediately.
    best = res["best_model"]
    model = tree.train_model(features, backend=best, **res["best_params"])
    tree.save_model(model, backend=best)
    print(f"[select] Retrained and saved winning model: {best}")


def run_validate_data() -> None:
    """Check raw CSV availability and create empty templates if needed."""
    loaders.validate_raw_data_files()


def run_sample_data() -> None:
    """Generate synthetic SAMPLE results plus optional context CSVs for demos."""
    from src.data import sample
    print("[sample-data] Generating synthetic sample data (NOT real)...")
    sample.write_sample_data()


def _print_top(team_stage: pd.DataFrame, n: int = 10) -> None:
    print("\nTop title contenders:")
    cols = ["team", "group", "prob_reach_sf", "prob_reach_final", "prob_champion"]
    top = team_stage.nlargest(n, "prob_champion")[cols].copy()
    for c in ("prob_reach_sf", "prob_reach_final", "prob_champion"):
        top[c] = (top[c] * 100).round(1).astype(str) + "%"
    print(top.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="2026 World Cup forecaster")
    parser.add_argument(
        "--mode", required=True,
        choices=[
            "validate-data",
            "sample-data",
            "train",
            "simulate",
            "full",
            "backtest",
            "tune",
            "select",
        ],
    )
    parser.add_argument("--n_simulations", type=int, default=config.QUICK_SIMULATIONS)
    parser.add_argument(
        "--simulation_seed",
        type=int,
        default=2026,
        help="Random seed for tournament Monte Carlo simulations.",
    )
    parser.add_argument(
        "--calibration",
        choices=["none", "sigmoid", "isotonic"],
        default="none",
        help="Optionally save calibrated classifier probabilities during train/full.",
    )
    parser.add_argument(
        "--odds_weight",
        type=float,
        default=0.0,
        help="Blend weight for optional data/raw/betting_odds.csv implied "
             "probabilities during simulation (0 disables odds blending).",
    )
    parser.add_argument(
        "--odds_method",
        choices=["shin", "basic"],
        default="shin",
        help="How to remove bookmaker margin from decimal odds.",
    )
    parser.add_argument(
        "--odds_blend",
        choices=["logarithmic", "linear"],
        default="logarithmic",
        help="How to blend model probabilities with no-vig market probabilities.",
    )
    parser.add_argument("--backtest_years", type=int, nargs="+", default=None,
                        help="World Cup years to backtest (default "
                             "2006 2010 2014 2018 2022).")
    parser.add_argument(
        "--model",
        choices=["best", "xgboost", "lightgbm", "catboost", "logistic", "elo"],
        default=DEFAULT_MODEL,
        help="Match model to drive the simulation. Tree backends are the "
             "primary models; 'best' uses the winner of `--mode select`; "
             "'logistic' is the baseline.",
    )
    args = parser.parse_args()

    if args.mode == "validate-data":
        run_validate_data()
    elif args.mode == "sample-data":
        run_sample_data()
    elif args.mode == "train":
        run_train(args.model, calibration=args.calibration)
    elif args.mode == "simulate":
        run_simulate(
            args.n_simulations,
            args.model,
            odds_weight=args.odds_weight,
            odds_method=args.odds_method,
            odds_blend=args.odds_blend,
            simulation_seed=args.simulation_seed,
        )
    elif args.mode == "backtest":
        run_backtest(args.model, years=args.backtest_years)
    elif args.mode == "tune":
        run_tune(args.model)
    elif args.mode == "select":
        run_select()
    elif args.mode == "full":
        run_train(args.model, calibration=args.calibration)
        run_simulate(
            args.n_simulations,
            args.model,
            odds_weight=args.odds_weight,
            odds_method=args.odds_method,
            odds_blend=args.odds_blend,
            simulation_seed=args.simulation_seed,
        )


if __name__ == "__main__":
    main()

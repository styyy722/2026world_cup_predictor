# 2026 FIFA World Cup Forecasting Model

A modular, Python-based forecasting system that predicts the win / draw / loss
probability of each match, simulates the full **2026 FIFA World Cup**, and
reports each team's probability of reaching every stage and winning the title.

The 2026 World Cup is the first **48-team** edition: **12 groups of 4**, with the
top 2 of each group plus the **8 best third-placed teams** advancing to a
**Round of 32** knockout bracket.

---

## Project purpose

* Predict per-match outcome probabilities from team strength + recent form.
* Run a Monte Carlo simulation of the whole tournament (default 10,000 runs).
* Output each team's group-placement, stage-progression, and title odds.
* Use gradient-boosted **tree models** (XGBoost / LightGBM / CatBoost) as the
  primary match model, with multinomial logistic regression and Elo as
  baselines for comparison.
* Provide a clean structure you can extend with richer data (player-level
  stats, injuries, betting odds, xG, market value).

---

## Project structure

```
.                      # repository root — all project files live here
  data/raw/            # manually supplied input CSVs; templates are created if missing
  data/processed/      # reserved for cached/processed data
  src/
    config.py          # paths + 2026 tournament constants
    data/              # loaders + template CSV generation
    features/          # match-level feature engineering
    models/            # tree_model.py (primary), baseline_logistic.py, elo_model.py
    simulation/        # match / group / knockout / tournament simulators
    evaluation/        # backtesting on past World Cups
    visualisation/     # chart generation
  outputs/tables/      # CSV outputs
  outputs/charts/      # PNG charts
  tests/               # pytest suite
  notebooks/           # exploratory analysis (yours to add)
  main.py              # CLI pipeline entry point
  dashboard.py         # Streamlit dashboard for output CSVs
  requirements.txt
```

---

## Data files required

The data pipeline is CSV-first. Place these files in `data/raw/`; no paid API
is required for the first version. If a file is missing, run
`python main.py --mode validate-data` and the project will create an empty CSV
template with the expected columns, then print clear instructions for where to
download or manually collect the data.

| File | Expected columns | Intended source |
|------|------------------|-----------------|
| `international_results.csv` | `date, home_team, away_team, home_score, away_score, tournament, neutral, country` | Kaggle [International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) |
| `fifa_rankings.csv` | `date, team, fifa_rank, fifa_points` | FIFA's [men's ranking page](https://inside.fifa.com/fifa-world-ranking/men?lv=true) for the latest snapshot, plus public historical rankings from Kaggle/GitHub |
| `elo_ratings.csv` | `date, team, elo_rating` | [World Football Elo Ratings](https://www.eloratings.net/) and/or a public Kaggle historical international football Elo dataset |
| `world_cup_2026_fixtures.csv` | `match_id, stage, group, date, team_a, team_b, venue, neutral` | FIFA's official [World Cup 2026 schedule](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/match-schedule-fixtures-results-teams-stadiums) |
| `world_cup_2026_groups.csv` | `group, team` | FIFA's official [World Cup 2026 final draw results](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/final-draw-results) |

API integrations should stay optional for a later version. The first version
should work from audited CSV files that can be committed, reviewed, and updated
manually.

Optional enhancement files can also be placed in `data/raw/`:

| File | Expected columns | Used for |
|------|------------------|----------|
| `team_context.csv` | `date, team, injured_players, suspended_players, squad_market_value_eur, xg_for_10, xg_against_10` | richer availability, market-value, and xG features |
| `betting_odds.csv` | `match_id, date, team_a, team_b, team_a_decimal_odds, draw_decimal_odds, team_b_decimal_odds, bookmaker` | optional bookmaker-implied probability blending |

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended.

---

## How to run

The `--model` flag selects which match model drives the simulation. The
default is **`xgboost`** (primary); `logistic` and `elo` are baselines.

```bash
# Validate raw-data files and create empty templates for any missing files
python main.py --mode validate-data

# Generate a synthetic SAMPLE dataset (incl. team_context.csv) to try the
# pipeline without real data (demo only - not real)
python main.py --mode sample-data

# Train the primary model (XGBoost) + the logistic baseline, and save both
python main.py --mode train                 # defaults to --model xgboost
python main.py --mode train --calibration sigmoid

# Train with a different tree backend
python main.py --mode train --model lightgbm
python main.py --mode train --model catboost

# Compare ALL three boosters (tuned) and select the best automatically
python main.py --mode select
# ...then drive everything with the winner
python main.py --mode full --model best --n_simulations 10000

# Simulate the tournament and export tables + charts (quick mode)
python main.py --mode simulate --n_simulations 10000 --model xgboost
python main.py --mode simulate --model xgboost --odds_weight 0.20

# Full pipeline: load -> features -> train -> predict -> simulate -> export
python main.py --mode full --n_simulations 10000 --model xgboost
python main.py --mode full --model xgboost --calibration isotonic --odds_weight 0.10

# Use a baseline instead of the tree model
python main.py --mode simulate --model logistic
python main.py --mode simulate --model elo

# Final, higher-precision run
python main.py --mode simulate --n_simulations 100000 --model xgboost

# Tune hyperparameters via walk-forward CV, save the best, and retrain
python main.py --mode tune --model xgboost

# Backtest: per-World-Cup + walk-forward CV over ALL history (vs baseline)
python main.py --mode backtest --model xgboost
```

`tune` saves the best hyperparameters to `models_store/<model>_best_params.json`;
`train` and `full` then load and apply them automatically (the run logs
"tuned params" when they do). `select` goes further: it tunes **all three
boosters** and records the winner in `models_store/selected_model.json`, which
`--model best` then uses.

Run the tests with:

```bash
pytest
```

Open the dashboard after simulation outputs exist:

```bash
streamlit run dashboard.py
```

---

## How the match model works

Three interchangeable models predict the three-way outcome
(`team_a_win`, `draw`, `team_b_win`). All share the same feature matrix and the
same probability interface (`src/models/common.py`), so the simulator treats
them identically.

1. **Gradient-boosted trees — primary** (`src/models/tree_model.py`)
   A multiclass gradient-boosting classifier with three interchangeable
   backends: **XGBoost**, **LightGBM**, and **CatBoost**. Trees capture
   non-linear interactions between features (e.g. how form matters more when
   teams are evenly matched) and need no feature scaling.

   **Why not just XGBoost?** No single booster is best on every dataset — they
   differ in tree-growth strategy (level-wise vs leaf-wise vs symmetric),
   regularisation, and how they handle small/noisy data. Rather than assume a
   winner, `--mode select` tunes all three and picks the one with the lowest
   walk-forward log loss on *your* data; `--model best` then uses it. XGBoost is
   only the default until a selection is made.

2. **Multinomial logistic regression — baseline** (`src/models/baseline_logistic.py`)
   A `StandardScaler` + softmax `LogisticRegression` pipeline. Linear, fast,
   and interpretable — a sensible benchmark the tree models should beat.

3. **Elo — baseline** (`src/models/elo_model.py`)
   The classic Elo expected-score formula plus a home-advantage term and a draw
   model that shrinks as the rating gap widens. No training required.

All models are trained on engineered match features (target `result`:
0 loss / 1 draw / 2 win from team A's perspective):

* **Strength:** Elo, FIFA rank/points and their differences.
* **Form (rolling 10 matches):** win rate, goals for/against, goal difference.
* **Team dynamics:** recent-form *momentum* (last-5 win rate minus the 10-match
  rate) and a signed win/loss *streak* — these capture short-term swings that
  the static Elo/FIFA snapshots miss.
* **Player/team availability (from optional `team_context.csv`):** number of
  absences (injuries + suspensions), squad market value, and recent xG balance.
* **Context:** rest-day gaps and neutral / World Cup / major-tournament flags.

Feature building lives in `src/features/build_features.py` and avoids look-ahead
leakage by only using each team's matches *before* kickoff. The availability and
dynamics features are populated when `team_context.csv` is present (real data
from Transfermarkt/FBref/Understat, or the synthetic `--mode sample-data` demo);
when it is absent they fall back to neutral values, so the pipeline still runs.

If `--calibration sigmoid` or `--calibration isotonic` is supplied during
`train` or `full`, the saved classifier is wrapped with scikit-learn probability
calibration before simulation. If `--odds_weight` is greater than zero and
`data/raw/betting_odds.csv` exists, model probabilities are blended with
normalised bookmaker-implied probabilities at that weight.

---

## How the tournament simulation works

`src/simulation/` runs a Monte Carlo simulation:

* **Group stage** — each group plays a round-robin (6 matches). Outcomes are
  sampled from the model probabilities; a plausible scoreline consistent with
  the sampled outcome is drawn so goals-for/against and goal difference are
  tracked. Teams are ranked by **points → goal difference → goals scored →
  random draw-of-lots** tie-break.
* **Qualification** — the top 2 of each of the 12 groups (24 teams) plus the
  **8 best third-placed teams** form the 32-team field.
* **Knockout** — qualifiers are placed into FIFA's published 2026 Round-of-32
  bracket slots, including the winner-v-third-place slot constraints
  (`1E v 3A/B/C/D/F`, etc.), then played as single elimination through R32 →
  R16 → QF → SF → Final. A regular-time draw is resolved by a
  **penalty-style** coin flip weighted toward the stronger team but damped
  toward 50/50.

Running thousands of tournaments and aggregating yields each team's stage and
title probabilities.

---

## Outputs

In `outputs/tables/`:

* **`team_stage_probabilities.csv`** — per team: group, expected group points,
  `prob_group_1st/2nd/3rd`, `prob_reach_r32/r16/qf/sf/final`, `prob_champion`.
* **`match_probabilities.csv`** — model WDL probabilities for each group fixture.
* **`simulation_summary.csv`** — per simulation: champion, runner-up,
  semi-finalists.
* **`backtest_summary.csv`** — accuracy / log loss / Brier per scope
  (`wc:<year>` World Cups and `walk_forward` over all history) and model.
* **`tuning_results_<model>.csv`** — every hyperparameter configuration tried
  during `--mode tune` / `--mode select`, with its walk-forward score.
* **`model_selection.csv`** — tuned walk-forward log loss / accuracy for each
  booster (XGBoost / LightGBM / CatBoost), best-first (written by `--mode select`).

In `outputs/charts/`:

* `champion_probabilities.png`
* `expected_group_points.png`
* `stage_progression_probabilities.png`
* `quarter_final_bracket.png` — poster-style projected knockout bracket built
  from the stage probabilities (most likely QF teams, finalists, and champion)

The Streamlit dashboard in `dashboard.py` reads these same output CSVs and gives
quick tabs for the title race, group outlook, match probabilities, and backtest
summary.

---

## Evaluation / backtesting

`src/evaluation/backtest.py` provides two time-respecting backtests:

1. **Per-World-Cup** — train on everything before a target World Cup, test on
   that tournament's matches (default 2006 / 2010 / 2014 / 2018 / 2022; override
   with `--backtest_years`). Realistic but a small test set.
2. **Walk-forward CV** — expanding-window time-series cross-validation
   (`TimeSeriesSplit`) over the **entire** match history, so every period after
   the first fold is scored out-of-sample. This is the metric used for tuning
   and the most reliable comparison between models.

Both report accuracy, multiclass log loss, multiclass Brier score, and a
calibration table, and compare the chosen tree model against the logistic
baseline.

### Tuning and model selection

`--mode tune` grid-searches one model's hyperparameters using the **walk-forward
log loss** (a proper scoring rule) as the objective, prints every config's
score, saves the best params, and retrains.

`--mode select` runs that search for **all three boosters** (XGBoost, LightGBM,
CatBoost), then picks the backend with the lowest walk-forward log loss and
records it in `models_store/selected_model.json`. Use `--model best` afterwards
to drive `train` / `simulate` / `full` with the winner. The per-booster grids
live in `_PARAM_GRIDS` and are easy to widen.

> Note: missing raw inputs now create empty templates only. Supply real
> historical data for meaningful model training and evaluation.

---

## Extension points

The main README improvements have been implemented as optional extension
points: richer team context features, betting-odds blending, calibrated
probabilities, FIFA-style knockout slots, and a Streamlit output dashboard. The
next natural upgrades are travel-distance features from venue coordinates,
model-specific early stopping with validation folds, and a goals-model backend
behind the same `MatchPredictor` interface.

---

## Assumptions

* Empty templates are placeholders only; replace them with real data for
  credible forecasts.
* World Cup group matches are treated as neutral-venue games.
* Third-place teams are assigned to FIFA-compatible R32 slots using deterministic
  backtracking across the published eligible-group constraints.
* Draw resolution in knockouts is a strength-weighted shootout approximation.

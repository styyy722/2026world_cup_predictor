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
worldcup_forecast/
  data/raw/            # input CSVs (auto-generated templates if missing)
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
  requirements.txt
```

---

## Data files required

Place these in `data/raw/`. **If any are missing, the pipeline auto-creates
self-consistent example templates** (a synthetic but plausible 48-team field)
and prints a message, so it runs out of the box. Replace them with real data.

| File | Schema |
|------|--------|
| `international_results.csv` | `date, home_team, away_team, home_score, away_score, tournament, neutral, country` |
| `fifa_rankings.csv` | `date, team, fifa_rank, fifa_points` |
| `elo_ratings.csv` | `date, team, elo_rating` |
| `world_cup_2026_fixtures.csv` | `match_id, stage, group, date, team_a, team_b, venue, neutral` |
| `world_cup_2026_groups.csv` | `group, team` |

Good public sources: international results datasets on Kaggle, the
[World Football Elo Ratings](https://eloratings.net), and FIFA's official
ranking exports.

---

## Installation

```bash
cd worldcup_forecast
pip install -r requirements.txt
```

Python 3.10+ recommended.

---

## How to run

The `--model` flag selects which match model drives the simulation. The
default is **`xgboost`** (primary); `logistic` and `elo` are baselines.

```bash
# Train the primary model (XGBoost) + the logistic baseline, and save both
python main.py --mode train                 # defaults to --model xgboost

# Train with a different tree backend
python main.py --mode train --model lightgbm
python main.py --mode train --model catboost   # requires `pip install catboost`

# Simulate the tournament and export tables + charts (quick mode)
python main.py --mode simulate --n_simulations 10000 --model xgboost

# Full pipeline: load -> features -> train -> predict -> simulate -> export
python main.py --mode full --n_simulations 10000 --model xgboost

# Use a baseline instead of the tree model
python main.py --mode simulate --model logistic
python main.py --mode simulate --model elo

# Final, higher-precision run
python main.py --mode simulate --n_simulations 100000 --model xgboost

# Backtest: compares the chosen tree model against the logistic baseline
python main.py --mode backtest --model xgboost
```

Run the tests with:

```bash
pytest
```

---

## How the match model works

Three interchangeable models predict the three-way outcome
(`team_a_win`, `draw`, `team_b_win`). All share the same feature matrix and the
same probability interface (`src/models/common.py`), so the simulator treats
them identically.

1. **Gradient-boosted trees — primary** (`src/models/tree_model.py`)
   A multiclass gradient-boosting classifier with three interchangeable
   backends: **XGBoost** (default), **LightGBM**, and **CatBoost**. Trees
   capture non-linear interactions between features (e.g. how form matters more
   when teams are evenly matched) and need no feature scaling. Defaults use
   shallow trees with mild regularisation to avoid overfitting the limited
   international match history; tune via keyword overrides to `train_model`.

2. **Multinomial logistic regression — baseline** (`src/models/baseline_logistic.py`)
   A `StandardScaler` + softmax `LogisticRegression` pipeline. Linear, fast,
   and interpretable — a sensible benchmark the tree models should beat.

3. **Elo — baseline** (`src/models/elo_model.py`)
   The classic Elo expected-score formula plus a home-advantage term and a draw
   model that shrinks as the rating gap widens. No training required.

All models are trained on engineered match features (target `result`:
0 loss / 1 draw / 2 win from team A's perspective): Elo, FIFA rank/points and
their differences, rolling 10-match form (win rate, goals for/against, goal
difference), and neutral / World Cup / major-tournament flags. Feature building
lives in `src/features/build_features.py` and avoids look-ahead leakage by only
using each team's matches *before* kickoff.

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
* **Knockout** — qualifiers are seeded into a balanced bracket by Elo (1-v-32,
  2-v-31, …) and played as single elimination through R32 → R16 → QF → SF →
  Final. A regular-time draw is resolved by a **penalty-style** coin flip
  weighted toward the stronger team but damped toward 50/50.

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
* **`backtest_summary.csv`** — accuracy / log loss / Brier per backtested year.

In `outputs/charts/`:

* `champion_probabilities.png`
* `expected_group_points.png`
* `stage_progression_probabilities.png`
* `quarter_final_bracket.png` — poster-style projected knockout bracket built
  from the stage probabilities (most likely QF teams, finalists, and champion)

---

## Evaluation / backtesting

`src/evaluation/backtest.py` trains only on matches *before* a target World Cup
and tests on that tournament's matches (2014, 2018, 2022), reporting accuracy,
multiclass log loss, multiclass Brier score, and a calibration table for each
model. By default it compares the chosen tree model against the logistic
baseline so you can see the lift. This time-respecting split avoids leakage.

> Note: the bundled **template** data is synthetic, so backtest numbers are
> illustrative. Supply real historical data for meaningful evaluation.

---

## How to improve the model later

The code is structured so you can extend it without rewrites:

* **Richer features** — add columns in `src/features/build_features.py`
  (player availability/injuries, squad market value, xG, rest days, travel).
* **Betting odds** — blend bookmaker-implied probabilities into the predictor.
* **Tune / extend models** — the tree backends in `src/models/tree_model.py`
  accept hyperparameter overrides; you could add early stopping, calibrated
  probabilities, or a bivariate-Poisson goals model behind the same
  `MatchPredictor` interface.
* **Official knockout seeding** — replace the Elo-seeded bracket in
  `simulate_knockout_stage.py` with FIFA's third-place placement table.
* **Calibration** — add probability calibration (e.g. isotonic) before
  simulating.
* **Dashboard** — a Streamlit app can consume the CSVs in `outputs/tables/`
  (intentionally not built in this first version).

---

## Assumptions

* Template/example data is synthetic and **not** an official draw or real
  ratings — replace it with real data for credible forecasts.
* World Cup group matches are treated as neutral-venue games.
* Knockout bracket seeding is by Elo rather than the official placement rules.
* Draw resolution in knockouts is a strength-weighted shootout approximation.

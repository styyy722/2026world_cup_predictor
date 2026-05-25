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
* Provide a clean baseline (Elo + logistic regression) that you can extend with
  richer data (player-level stats, injuries, betting odds, xG, market value).

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
    models/            # elo_model.py, baseline_logistic.py
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

```bash
# Train the logistic model on historical results and save it
python main.py --mode train

# Simulate the tournament and export tables + charts (quick mode)
python main.py --mode simulate --n_simulations 10000

# Full pipeline: load -> features -> train -> predict -> simulate -> export
python main.py --mode full --n_simulations 10000

# Use the Elo baseline instead of the logistic model
python main.py --mode simulate --model elo

# Final, higher-precision run
python main.py --mode simulate --n_simulations 100000

# Backtest the match model on the 2014 / 2018 / 2022 World Cups
python main.py --mode backtest
```

Run the tests with:

```bash
pytest
```

---

## How the match model works

Two interchangeable models predict the three-way outcome
(`team_a_win`, `draw`, `team_b_win`):

1. **Elo baseline** (`src/models/elo_model.py`)
   Uses the classic Elo expected-score formula on the two teams' ratings, adds
   a home-advantage term for non-neutral games, and splits probability into
   win/draw/loss with a draw model that shrinks as the rating gap widens.

2. **Multinomial logistic regression** (`src/models/baseline_logistic.py`)
   Trained on engineered match features (encoded `result`: 0 loss / 1 draw /
   2 win from team A's perspective). Features include Elo, FIFA rank/points,
   their differences, rolling 10-match form (win rate, goals for/against, goal
   difference), and flags for neutral venue / World Cup / major tournament.
   Feature building lives in `src/features/build_features.py` and avoids
   look-ahead leakage by only using each team's matches *before* kickoff.

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

---

## Evaluation / backtesting

`src/evaluation/backtest.py` trains only on matches *before* a target World Cup
and tests on that tournament's matches (2014, 2018, 2022), reporting accuracy,
multiclass log loss, multiclass Brier score, and a calibration table. This
time-respecting split avoids leakage.

> Note: the bundled **template** data is synthetic, so backtest numbers are
> illustrative. Supply real historical data for meaningful evaluation.

---

## How to improve the model later

The code is structured so you can extend it without rewrites:

* **Richer features** — add columns in `src/features/build_features.py`
  (player availability/injuries, squad market value, xG, rest days, travel).
* **Betting odds** — blend bookmaker-implied probabilities into the predictor.
* **Better match model** — swap the logistic model for gradient boosting or a
  bivariate-Poisson goals model (the `MatchPredictor` interface stays the same).
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

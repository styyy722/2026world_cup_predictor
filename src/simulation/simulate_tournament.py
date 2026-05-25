"""Full-tournament Monte Carlo simulation and aggregation.

Runs the group stage + knockout bracket many times and aggregates each team's
probability of every group placement and knockout stage, plus per-simulation
summaries (champion / runner-up / semi-finalists).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from . import simulate_group_stage as groups_sim
from . import simulate_knockout_stage as ko

# Maps a furthest-stage label to an ordinal for cumulative "reached" logic.
_STAGE_RANK = {"R32": 0, "R16": 1, "QF": 2, "SF": 3, "F": 4, "CHAMPION": 5}


def run_simulations(predictor, groups: pd.DataFrame, fixtures: pd.DataFrame,
                    n_simulations: int = 10_000, seed: int = 2026) -> dict:
    """Run ``n_simulations`` full tournaments.

    Returns a dict of two DataFrames:
        ``team_stage`` -> per-team aggregated probabilities and expected points
        ``summary``    -> per-simulation champion / runner-up / semi-finalists
    """
    rng = np.random.default_rng(seed)

    all_teams = list(groups["team"].unique())
    team_group = dict(zip(groups["team"], groups["group"]))

    # Accumulators.
    cnt = lambda: defaultdict(int)
    first = cnt(); second = cnt(); third = cnt()
    points_sum = defaultdict(float)
    reach = {s: cnt() for s in ["R32", "R16", "QF", "SF", "F", "CHAMPION"]}

    summary_rows = []

    for sim_id in range(n_simulations):
        gres = groups_sim.simulate_all_groups(groups, fixtures, predictor, rng)

        # Group placements + expected points.
        for g, ranked in gres["standings"].items():
            for s in ranked:
                points_sum[s["team"]] += s["points"]
            first[ranked[0]["team"]] += 1
            second[ranked[1]["team"]] += 1
            third[ranked[2]["team"]] += 1

        # Knockout field = 1st + 2nd of each group + best-8 thirds.
        qualified = gres["winners"] + gres["runners_up"] + gres["qualified_thirds"]
        bracket = ko.build_r32_bracket(qualified, predictor)
        kres = ko.simulate_knockout(bracket, predictor, rng)

        # Cumulative stage-reach tallies.
        for team, furthest in kres["reached"].items():
            r = _STAGE_RANK[furthest]
            if r >= 0: reach["R32"][team] += 1
            if r >= 1: reach["R16"][team] += 1
            if r >= 2: reach["QF"][team] += 1
            if r >= 3: reach["SF"][team] += 1
            if r >= 4: reach["F"][team] += 1
            if r >= 5: reach["CHAMPION"][team] += 1

        summary_rows.append({
            "simulation_id": sim_id,
            "champion": kres["champion"],
            "runner_up": kres["runner_up"],
            "semi_finalists": "|".join(sorted(kres["semi_finalists"])),
        })

    n = float(n_simulations)
    team_rows = []
    for team in all_teams:
        team_rows.append({
            "team": team,
            "group": team_group.get(team, ""),
            "expected_group_points": points_sum[team] / n,
            "prob_group_1st": first[team] / n,
            "prob_group_2nd": second[team] / n,
            "prob_group_3rd": third[team] / n,
            "prob_reach_r32": reach["R32"][team] / n,
            "prob_reach_r16": reach["R16"][team] / n,
            "prob_reach_qf": reach["QF"][team] / n,
            "prob_reach_sf": reach["SF"][team] / n,
            "prob_reach_final": reach["F"][team] / n,
            "prob_champion": reach["CHAMPION"][team] / n,
        })

    team_stage = pd.DataFrame(team_rows).sort_values(
        "prob_champion", ascending=False).reset_index(drop=True)
    summary = pd.DataFrame(summary_rows)
    return {"team_stage": team_stage, "summary": summary}


def fixture_match_probabilities(predictor, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Model probabilities for each known (group-stage) fixture.

    Knockout matchups are not fixed in advance (they depend on simulated group
    outcomes), so only the deterministic group fixtures are listed here.
    """
    rows = []
    for m in fixtures.itertuples(index=False):
        pa, pd_, pb = predictor.proba(m.team_a, m.team_b, neutral=bool(m.neutral))
        rows.append({
            "match_id": m.match_id,
            "stage": m.stage,
            "team_a": m.team_a,
            "team_b": m.team_b,
            "team_a_win_prob": round(pa, 4),
            "draw_prob": round(pd_, 4),
            "team_b_win_prob": round(pb, 4),
        })
    return pd.DataFrame(rows)

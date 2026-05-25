"""Group-stage simulation and ranking for the 48-team / 12-group format.

Each group plays a round-robin (6 matches). Teams are ranked by:
    1. points (win=3, draw=1, loss=0)
    2. goal difference
    3. goals scored
    4. random tie-break (stands in for the official drawing of lots)

The top 2 of every group advance, and the best 8 of the 12 third-placed teams
also advance, forming the 32-team Round of 32.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from . import simulate_match as sm


def _new_stat() -> dict:
    return {"points": 0, "gf": 0, "ga": 0, "gd": 0, "played": 0}


def simulate_group(group: str, fixtures: pd.DataFrame,
                   predictor, rng: np.random.Generator) -> list[dict]:
    """Simulate one group and return its ranked standings.

    ``fixtures`` are the group matches for this group only. Returns a list of
    per-team standing dicts ordered best-to-worst, each tagged with ``rank``
    (1-based) and the ``group`` name.
    """
    stats: dict[str, dict] = {}

    for m in fixtures.itertuples(index=False):
        a, b = m.team_a, m.team_b
        stats.setdefault(a, _new_stat())
        stats.setdefault(b, _new_stat())

        probs = predictor.proba(a, b, neutral=bool(m.neutral))
        ga, gb, outcome = sm.simulate_scoreline(probs, rng)

        stats[a]["gf"] += ga; stats[a]["ga"] += gb
        stats[b]["gf"] += gb; stats[b]["ga"] += ga
        stats[a]["played"] += 1; stats[b]["played"] += 1

        if outcome == sm.TEAM_A:
            stats[a]["points"] += config.POINTS_WIN
            stats[b]["points"] += config.POINTS_LOSS
        elif outcome == sm.TEAM_B:
            stats[b]["points"] += config.POINTS_WIN
            stats[a]["points"] += config.POINTS_LOSS
        else:
            stats[a]["points"] += config.POINTS_DRAW
            stats[b]["points"] += config.POINTS_DRAW

    for s in stats.values():
        s["gd"] = s["gf"] - s["ga"]

    ranked = rank_group(stats, rng)
    out = []
    for rank, team in enumerate(ranked, start=1):
        row = {"team": team, "group": group, "rank": rank, **stats[team]}
        out.append(row)
    return out


def rank_group(stats: dict[str, dict], rng: np.random.Generator) -> list[str]:
    """Rank teams within a group, returning team names best-to-worst.

    Sort key: points desc, goal difference desc, goals for desc, random.
    A random jitter implements the "drawing of lots" final tie-break.
    """
    teams = list(stats.keys())
    jitter = {t: rng.random() for t in teams}
    return sorted(
        teams,
        key=lambda t: (stats[t]["points"], stats[t]["gd"], stats[t]["gf"], jitter[t]),
        reverse=True,
    )


def select_best_third_placed(third_placed: list[dict],
                             rng: np.random.Generator,
                             n: int = config.N_THIRD_PLACE_QUALIFIERS) -> list[dict]:
    """Pick the best ``n`` third-placed teams across all groups.

    ``third_placed`` is a list of standing dicts (one per group's 3rd team).
    Ranked by the same criteria as within a group.
    """
    jitter = {i: rng.random() for i in range(len(third_placed))}
    ordered = sorted(
        range(len(third_placed)),
        key=lambda i: (
            third_placed[i]["points"],
            third_placed[i]["gd"],
            third_placed[i]["gf"],
            jitter[i],
        ),
        reverse=True,
    )
    return [third_placed[i] for i in ordered[:n]]


def simulate_all_groups(groups: pd.DataFrame, fixtures: pd.DataFrame,
                        predictor, rng: np.random.Generator) -> dict:
    """Simulate every group.

    Returns a dict with:
        ``standings``  -> {group: [ranked standing dicts]}
        ``winners``    -> list of 1st-place standing dicts
        ``runners_up`` -> list of 2nd-place standing dicts
        ``thirds``     -> list of all 3rd-place standing dicts
        ``qualified_thirds`` -> best-8 third-place standing dicts
    """
    group_names = list(groups["group"].unique())
    standings = {}
    winners, runners_up, thirds = [], [], []

    for g in group_names:
        gfix = fixtures[fixtures["group"] == g]
        ranked = simulate_group(g, gfix, predictor, rng)
        standings[g] = ranked
        winners.append(ranked[0])
        runners_up.append(ranked[1])
        thirds.append(ranked[2])

    qualified_thirds = select_best_third_placed(thirds, rng)
    return {
        "standings": standings,
        "winners": winners,
        "runners_up": runners_up,
        "thirds": thirds,
        "qualified_thirds": qualified_thirds,
    }

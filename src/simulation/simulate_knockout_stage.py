"""Knockout-stage simulation: Round of 32 down to the final.

The 32 qualifiers (12 group winners + 12 runners-up + 8 best third-placed
teams) are seeded into a single-elimination bracket by Elo strength using the
standard 1-v-32, 2-v-31 ... pattern, which produces a balanced bracket without
needing the official (and complex) third-place placement table.
"""
from __future__ import annotations

import numpy as np

from .. import config
from . import simulate_match as sm


def build_r32_bracket(qualified: list[dict], predictor) -> list[str]:
    """Seed the 32 qualifiers into bracket order.

    ``qualified`` is a list of standing dicts (each has a ``team``). Returns a
    flat list of 32 team names where adjacent pairs (0,1), (2,3), ... are the
    Round-of-32 matchups. Seeding is by Elo: strongest plays weakest, etc.
    """
    teams = [q["team"] for q in qualified]
    teams_sorted = sorted(teams, key=lambda t: predictor.elo(t), reverse=True)
    n = len(teams_sorted)
    bracket = []
    for i in range(n // 2):
        bracket.append(teams_sorted[i])
        bracket.append(teams_sorted[n - 1 - i])
    return bracket


def simulate_knockout(bracket: list[str], predictor,
                      rng: np.random.Generator) -> dict:
    """Run the single-elimination bracket to completion.

    Returns a dict with:
        ``champion``        -> team name
        ``runner_up``       -> team name (loser of the final)
        ``semi_finalists``  -> list of the 4 SF teams
        ``reached``         -> {team: furthest stage label reached}
    Stage labels follow ``config.KNOCKOUT_STAGES`` plus ``CHAMPION``.
    """
    reached: dict[str, str] = {}
    current = list(bracket)
    semi_finalists: list[str] = []
    finalists: list[str] = []

    for stage in config.KNOCKOUT_STAGES:  # ["R32","R16","QF","SF","F"]
        # Everyone still in the bracket has reached this stage.
        for t in current:
            reached[t] = stage
        if stage == "SF":
            semi_finalists = list(current)
        if stage == "F":
            finalists = list(current)

        winners = []
        for i in range(0, len(current), 2):
            a, b = current[i], current[i + 1]
            probs = predictor.proba(a, b, neutral=True)
            winner = sm.simulate_knockout_winner(
                a, b, probs, predictor.elo(a), predictor.elo(b), rng
            )
            winners.append(winner)
        current = winners

    champion = current[0]
    reached[champion] = "CHAMPION"
    runner_up = finalists[0] if finalists[1] == champion else finalists[1]

    return {
        "champion": champion,
        "runner_up": runner_up,
        "semi_finalists": semi_finalists,
        "reached": reached,
    }

"""Knockout-stage simulation: Round of 32 down to the final.

The 32 qualifiers (12 group winners + 12 runners-up + 8 best third-placed
teams) are placed into FIFA's published 2026 Round-of-32 bracket slots. Third
placed teams are assigned to the eligible winner-v-third slots using the
published slot constraints (for example, ``1E`` plays a third-placed team from
``A/B/C/D/F``) and a deterministic backtracking assignment.
"""
from __future__ import annotations

import numpy as np

from .. import config
from . import simulate_match as sm

# Ordered so the generic pairwise simulator reproduces the published routes:
# R16 M89: W74-W77, M90: W73-W75, M93: W83-W84, M94: W81-W82,
# M91: W76-W78, M92: W79-W80, M95: W86-W88, M96: W85-W87.
R32_SLOT_ORDER = (
    (74, "1E", "3ABCDF"),
    (77, "1I", "3CDFGH"),
    (73, "2A", "2B"),
    (75, "1F", "2C"),
    (83, "2K", "2L"),
    (84, "1H", "2J"),
    (81, "1D", "3BEFIJ"),
    (82, "1G", "3AEHIJ"),
    (76, "1C", "2F"),
    (78, "2E", "2I"),
    (79, "1A", "3CEFHI"),
    (80, "1L", "3EHIJK"),
    (86, "1J", "2H"),
    (88, "2D", "2G"),
    (85, "1B", "3EFGIJ"),
    (87, "1K", "3DEIJL"),
)


def build_r32_bracket(qualified: list[dict], predictor,
                      group_results: dict | None = None) -> list[str]:
    """Seed the 32 qualifiers into bracket order.

    ``qualified`` is a list of standing dicts (each has a ``team``). Returns a
    flat list of 32 team names where adjacent pairs (0,1), (2,3), ... are the
    Round-of-32 matchups. If group-stage results are provided, FIFA's published
    2026 bracket slots are used. Otherwise this falls back to Elo seeding for
    older tests and ad hoc simulations.
    """
    if group_results is not None:
        return build_official_r32_bracket(group_results)

    teams = [q["team"] for q in qualified]
    teams_sorted = sorted(teams, key=lambda t: predictor.elo(t), reverse=True)
    n = len(teams_sorted)
    bracket = []
    for i in range(n // 2):
        bracket.append(teams_sorted[i])
        bracket.append(teams_sorted[n - 1 - i])
    return bracket


def r32_pairings_from_group_results(group_results: dict) -> list[dict]:
    """Return FIFA-style Round-of-32 pairings from simulated group results."""
    team_by_slot = _slot_team_map(group_results)
    pairings = []
    for match_id, slot_a, slot_b in R32_SLOT_ORDER:
        pairings.append({
            "match_id": match_id,
            "slot_a": slot_a,
            "slot_b": slot_b,
            "team_a": team_by_slot[slot_a],
            "team_b": team_by_slot[slot_b],
        })
    return pairings


def build_official_r32_bracket(group_results: dict) -> list[str]:
    """Build a flat R32 bracket from FIFA's published 2026 slot constraints."""
    bracket: list[str] = []
    for pairing in r32_pairings_from_group_results(group_results):
        bracket.extend([pairing["team_a"], pairing["team_b"]])
    return bracket


def _slot_team_map(group_results: dict) -> dict[str, str]:
    standings = group_results["standings"]
    team_by_group_rank: dict[tuple[str, int], str] = {}
    for group, ranked in standings.items():
        for row in ranked:
            team_by_group_rank[(str(group), int(row["rank"]))] = row["team"]

    third_assignments = _assign_third_place_slots(
        group_results["qualified_thirds"],
        [slot for _, _, slot in R32_SLOT_ORDER if slot.startswith("3")],
    )

    out: dict[str, str] = {}
    for _, slot_a, slot_b in R32_SLOT_ORDER:
        for slot in (slot_a, slot_b):
            if slot in out:
                continue
            if slot.startswith(("1", "2")):
                out[slot] = team_by_group_rank[(slot[1], int(slot[0]))]
            elif slot.startswith("3"):
                group = third_assignments[slot]
                out[slot] = team_by_group_rank[(group, 3)]
            else:  # pragma: no cover - constants guard this
                raise ValueError(f"Unknown knockout slot label: {slot}")
    return out


def _assign_third_place_slots(qualified_thirds: list[dict],
                              third_slots: list[str]) -> dict[str, str]:
    """Assign qualified third-place groups to compatible R32 slots.

    FIFA publishes each winner-v-third slot as a set of eligible third-place
    groups. There are many possible combinations of advancing third-place
    groups, so this uses deterministic backtracking to find a complete
    one-to-one assignment while preferring higher-ranked third-place teams.
    """
    third_rank = {str(row["group"]): i for i, row in enumerate(qualified_thirds)}
    qualified_groups = set(third_rank)
    candidates = {
        slot: sorted(
            [g for g in slot[1:] if g in qualified_groups],
            key=lambda g: third_rank[g],
        )
        for slot in third_slots
    }
    if any(not groups for groups in candidates.values()):
        missing = [slot for slot, groups in candidates.items() if not groups]
        raise ValueError(f"No compatible qualified third-place team for {missing}")

    slot_order = sorted(third_slots, key=lambda slot: (len(candidates[slot]), slot))

    def search(i: int, used: set[str], assigned: dict[str, str]) -> dict[str, str] | None:
        if i == len(slot_order):
            return assigned
        slot = slot_order[i]
        for group in candidates[slot]:
            if group in used:
                continue
            nxt = dict(assigned)
            nxt[slot] = group
            found = search(i + 1, used | {group}, nxt)
            if found is not None:
                return found
        return None

    assigned = search(0, set(), {})
    if assigned is None:
        raise ValueError(
            "Could not assign qualified third-place teams to R32 slots. "
            "Check the group-results structure and FIFA slot constraints."
        )
    return assigned


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

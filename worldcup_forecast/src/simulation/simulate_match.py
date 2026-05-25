"""Single-match simulation primitives.

Group matches need a scoreline (for goal difference tie-breaks); knockout
matches only need a single winner (with a penalty-style tie-break on draws).
"""
from __future__ import annotations

import numpy as np

# Outcome labels.
TEAM_A = "a"
DRAW = "draw"
TEAM_B = "b"


def simulate_outcome(probs: tuple[float, float, float],
                     rng: np.random.Generator) -> str:
    """Sample win/draw/loss from (P_a_win, P_draw, P_b_win)."""
    p_a, p_d, p_b = probs
    total = p_a + p_d + p_b
    # Guard against tiny numerical drift so probabilities sum to 1.
    r = rng.random() * total
    if r < p_a:
        return TEAM_A
    if r < p_a + p_d:
        return DRAW
    return TEAM_B


def _expected_goals(probs: tuple[float, float, float]) -> tuple[float, float]:
    """Derive per-team expected goals from outcome probabilities.

    Heuristic: a baseline of ~1.3 goals each, tilted by the win-probability
    margin so the favourite scores more on average. Only used to make sampled
    scorelines plausible; the *outcome* itself comes from the model.
    """
    p_a, _p_d, p_b = probs
    margin = p_a - p_b  # in [-1, 1]
    base = 1.3
    lam_a = max(0.2, base * (1 + 0.6 * margin))
    lam_b = max(0.2, base * (1 - 0.6 * margin))
    return lam_a, lam_b


def simulate_scoreline(probs: tuple[float, float, float],
                       rng: np.random.Generator,
                       max_tries: int = 20) -> tuple[int, int, str]:
    """Simulate a group-match scoreline consistent with a sampled outcome.

    Returns ``(goals_a, goals_b, outcome)``. We first sample the outcome from
    the model probabilities, then draw Poisson scorelines until one matches
    that outcome (falling back to a minimal consistent scoreline).
    """
    outcome = simulate_outcome(probs, rng)
    lam_a, lam_b = _expected_goals(probs)

    for _ in range(max_tries):
        ga = int(rng.poisson(lam_a))
        gb = int(rng.poisson(lam_b))
        if outcome == TEAM_A and ga > gb:
            return ga, gb, outcome
        if outcome == TEAM_B and gb > ga:
            return ga, gb, outcome
        if outcome == DRAW and ga == gb:
            return ga, gb, outcome

    # Fallback: construct a minimal scoreline matching the outcome.
    if outcome == TEAM_A:
        return 1, 0, outcome
    if outcome == TEAM_B:
        return 0, 1, outcome
    return 1, 1, outcome


def simulate_knockout_winner(team_a: str, team_b: str,
                             probs: tuple[float, float, float],
                             elo_a: float, elo_b: float,
                             rng: np.random.Generator) -> str:
    """Return the winning team name for a knockout match.

    A draw in regular time is resolved by a penalty-style coin-flip weighted by
    relative Elo strength (so the stronger team is a slight favourite, but
    upsets happen, as in real shootouts).
    """
    outcome = simulate_outcome(probs, rng)
    if outcome == TEAM_A:
        return team_a
    if outcome == TEAM_B:
        return team_b

    # Draw -> penalty shootout. Convert Elo gap into a shootout win prob,
    # damped toward 50/50 since shootouts are near-random.
    gap = elo_a - elo_b
    p_a = 1.0 / (1.0 + 10 ** (-gap / 1200.0))  # large scale -> close to 0.5
    return team_a if rng.random() < p_a else team_b

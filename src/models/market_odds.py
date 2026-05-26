"""Utilities for turning betting-market odds into usable probabilities.

The goal is to use the market as a calibrated prior, not to treat bookmaker
prices as raw truth. Decimal odds are first converted to implied probabilities,
the overround is removed, and rows from multiple books/snapshots are aggregated
into a consensus estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _normalise(values: np.ndarray) -> tuple[float, ...]:
    total = float(values.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Probabilities must have a positive finite sum.")
    return tuple((values / total).astype(float))


def _decimal_odds_array(decimal_odds) -> np.ndarray:
    odds = np.asarray(decimal_odds, dtype=float)
    if odds.ndim != 1 or len(odds) < 2:
        raise ValueError("Expected a one-dimensional odds vector.")
    if np.any(~np.isfinite(odds)) or np.any(odds <= 1.0):
        raise ValueError("Decimal odds must be finite values greater than 1.")
    return odds


def basic_no_vig_probabilities(decimal_odds) -> tuple[float, ...]:
    """Convert decimal odds to probabilities with proportional vig removal."""
    odds = _decimal_odds_array(decimal_odds)
    implied = 1.0 / odds
    return _normalise(implied)


def shin_no_vig_probabilities(
    decimal_odds,
    *,
    max_iter: int = 1000,
    tol: float = 1e-12,
) -> tuple[float, ...]:
    """Convert decimal odds using Shin's favourite-longshot adjustment.

    Shin's method estimates no-vig probabilities while allowing for bias caused
    by informed money and bookmaker risk management. For two-way or no-margin
    markets the method collapses back to basic normalisation.
    """
    odds = _decimal_odds_array(decimal_odds)
    implied = 1.0 / odds
    booksum = float(implied.sum())
    if len(implied) < 3 or booksum <= 1.0 + tol:
        return basic_no_vig_probabilities(odds)

    z = 0.0
    denom = len(implied) - 2
    for _ in range(max_iter):
        root_terms = np.sqrt(z * z + 4.0 * (1.0 - z) * (implied ** 2) / booksum)
        next_z = (float(root_terms.sum()) - 2.0) / denom
        next_z = float(np.clip(next_z, 0.0, 0.999999))
        if abs(next_z - z) < tol:
            z = next_z
            break
        z = next_z

    probs = (
        np.sqrt(z * z + 4.0 * (1.0 - z) * (implied ** 2) / booksum) - z
    ) / (2.0 * (1.0 - z))
    return _normalise(probs)


def no_vig_probabilities(decimal_odds, method: str = "shin") -> tuple[float, ...]:
    """Convert decimal odds into fair probabilities using the chosen method."""
    method = method.lower()
    if method == "basic":
        return basic_no_vig_probabilities(decimal_odds)
    if method == "shin":
        try:
            return shin_no_vig_probabilities(decimal_odds)
        except (FloatingPointError, ValueError, ZeroDivisionError):
            return basic_no_vig_probabilities(decimal_odds)
    raise ValueError(f"Unknown odds conversion method '{method}'.")


def blend_probabilities(
    model_probs: tuple[float, ...],
    market_probs: tuple[float, ...],
    *,
    weight: float,
    method: str = "logarithmic",
) -> tuple[float, ...]:
    """Blend model and market probabilities.

    The default logarithmic pool treats the betting market as another
    probabilistic expert and preserves a proper probability vector.
    """
    w = float(np.clip(weight, 0.0, 1.0))
    model = np.asarray(model_probs, dtype=float)
    market = np.asarray(market_probs, dtype=float)
    if model.shape != market.shape:
        raise ValueError("Model and market probabilities must have the same shape.")
    if w == 0.0:
        return tuple(model.astype(float))
    if w == 1.0:
        return _normalise(market)

    method = method.lower()
    if method == "linear":
        return _normalise((1.0 - w) * model + w * market)
    if method in {"log", "logarithmic"}:
        eps = 1e-12
        pooled = np.exp(
            (1.0 - w) * np.log(np.clip(model, eps, 1.0))
            + w * np.log(np.clip(market, eps, 1.0))
        )
        return _normalise(pooled)
    raise ValueError(f"Unknown odds blend method '{method}'.")


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "closing"}


def _candidate_rows(
    odds: pd.DataFrame,
    team_a: str,
    team_b: str,
    match_id: int | str | None,
) -> pd.DataFrame:
    rows = pd.DataFrame()
    if match_id is not None and "match_id" in odds.columns:
        rows = odds[odds["match_id"].astype(str) == str(match_id)].copy()
    if rows.empty:
        direct = (odds["team_a"] == team_a) & (odds["team_b"] == team_b)
        reverse = (odds["team_a"] == team_b) & (odds["team_b"] == team_a)
        rows = odds[direct | reverse].copy()
    if rows.empty:
        return rows

    if "is_closing" in rows.columns:
        closing = rows[rows["is_closing"].apply(_truthy)].copy()
        if not closing.empty:
            rows = closing

    if "snapshot_time" in rows.columns:
        rows["_snapshot_time"] = pd.to_datetime(rows["snapshot_time"], errors="coerce")
        if rows["_snapshot_time"].notna().any():
            rows = rows.sort_values("_snapshot_time")
            if "bookmaker" in rows.columns:
                rows = rows.groupby("bookmaker", as_index=False).tail(1)
            else:
                rows = rows.tail(1)
    return rows


def consensus_market_probabilities(
    odds: pd.DataFrame,
    team_a: str,
    team_b: str,
    *,
    match_id: int | str | None = None,
    method: str = "shin",
) -> tuple[float, float, float] | None:
    """Return median no-vig WDL probabilities across available odds rows."""
    if odds is None or odds.empty:
        return None
    required = {"team_a", "team_b", "team_a_decimal_odds",
                "draw_decimal_odds", "team_b_decimal_odds"}
    if not required.issubset(odds.columns):
        return None

    rows = _candidate_rows(odds, team_a, team_b, match_id)
    if rows.empty:
        return None

    converted: list[tuple[float, float, float]] = []
    for row in rows.itertuples(index=False):
        row_dict = row._asdict()
        dec = [
            row_dict["team_a_decimal_odds"],
            row_dict["draw_decimal_odds"],
            row_dict["team_b_decimal_odds"],
        ]
        try:
            probs = no_vig_probabilities(dec, method=method)
        except (TypeError, ValueError):
            continue
        if str(row_dict["team_a"]) == team_b and str(row_dict["team_b"]) == team_a:
            probs = (probs[2], probs[1], probs[0])
        converted.append(tuple(float(p) for p in probs))

    if not converted:
        return None
    consensus = np.median(np.asarray(converted, dtype=float), axis=0)
    return _normalise(consensus)  # type: ignore[return-value]

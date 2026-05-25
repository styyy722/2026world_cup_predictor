"""Chart generation for tournament outputs.

Produces clean, readable matplotlib PNGs from the aggregated team-stage table:
* champion_probabilities.png       - top teams by title probability
* expected_group_points.png        - expected group points per team
* stage_progression_probabilities.png - progression curve for top teams
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for servers / CI
import matplotlib.pyplot as plt
import pandas as pd

from .. import config

_STAGE_COLS = [
    ("prob_reach_r32", "R32"),
    ("prob_reach_r16", "R16"),
    ("prob_reach_qf", "QF"),
    ("prob_reach_sf", "SF"),
    ("prob_reach_final", "Final"),
    ("prob_champion", "Champion"),
]


def plot_champion_probabilities(team_stage: pd.DataFrame, top_n: int = 15,
                                path: Path | None = None) -> Path:
    path = path or (config.CHARTS_DIR / "champion_probabilities.png")
    df = team_stage.nlargest(top_n, "prob_champion").iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(df["team"], df["prob_champion"] * 100, color="#1f77b4")
    ax.set_xlabel("Championship probability (%)")
    ax.set_title(f"Top {top_n} title contenders - 2026 World Cup")
    for y, v in enumerate(df["prob_champion"] * 100):
        ax.text(v + 0.1, y, f"{v:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_expected_group_points(team_stage: pd.DataFrame, top_n: int = 20,
                               path: Path | None = None) -> Path:
    path = path or (config.CHARTS_DIR / "expected_group_points.png")
    df = team_stage.nlargest(top_n, "expected_group_points").iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(df["team"], df["expected_group_points"], color="#2ca02c")
    ax.set_xlabel("Expected group-stage points")
    ax.set_title(f"Expected group points - top {top_n} teams")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_stage_progression(team_stage: pd.DataFrame, top_n: int = 8,
                           path: Path | None = None) -> Path:
    path = path or (config.CHARTS_DIR / "stage_progression_probabilities.png")
    df = team_stage.nlargest(top_n, "prob_champion")
    stages = [label for _, label in _STAGE_COLS]
    cols = [c for c, _ in _STAGE_COLS]
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in df.iterrows():
        ax.plot(stages, [row[c] * 100 for c in cols], marker="o", label=row["team"])
    ax.set_ylabel("Probability (%)")
    ax.set_title(f"Stage progression probabilities - top {top_n} teams")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def generate_all_charts(team_stage: pd.DataFrame) -> list[Path]:
    """Render every chart and return the written paths."""
    config.ensure_dirs()
    return [
        plot_champion_probabilities(team_stage),
        plot_expected_group_points(team_stage),
        plot_stage_progression(team_stage),
    ]

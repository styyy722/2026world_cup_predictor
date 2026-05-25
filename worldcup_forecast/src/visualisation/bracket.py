"""Quarter-final bracket chart, styled after a classic tournament poster.

Builds a "most likely" knockout bracket from the aggregated simulation output
(`team_stage_probabilities.csv`): the 8 teams most likely to reach the
quarter-finals are placed at the QF slots, the two most likely finalists sit at
the finalist nodes, and the highest title probability team is named at the
centre FINAL node. Each node is annotated with the relevant probability.

The layout mirrors a QF poster: QF (4 per side) -> SF (2 per side) ->
finalist (1 per side) -> FINAL in the centre. There are no flag images, so each
team is drawn as a circular node with a short code and full name.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

from .. import config

# Poster-style palette (deep magenta background, white lines, light accents).
BG = "#8a1f4a"
LINE = "#ffffff"
NODE_FACE = "#ffffff"
NODE_EDGE = "#ffd6e6"
TEXT_DARK = "#7a1340"
ACCENT = "#ffc1da"

# Hand-curated 3-letter codes; falls back to first 3 letters for others.
_CODES = {
    "Argentina": "ARG", "France": "FRA", "Brazil": "BRA", "England": "ENG",
    "Spain": "ESP", "Germany": "GER", "Portugal": "POR", "Netherlands": "NED",
    "Belgium": "BEL", "Croatia": "CRO", "Italy": "ITA", "Uruguay": "URU",
    "Switzerland": "SUI", "Denmark": "DEN", "Morocco": "MAR", "Mexico": "MEX",
    "USA": "USA", "Colombia": "COL", "Japan": "JPN", "Senegal": "SEN",
    "Austria": "AUT", "Norway": "NOR", "Serbia": "SRB", "Sweden": "SWE",
    "Ukraine": "UKR", "Nigeria": "NGA", "Ecuador": "ECU", "Poland": "POL",
}


def _code(team: str) -> str:
    return _CODES.get(team, team[:3].upper())


def _draw_node(ax, x, y, team, prob, r=0.42, prob_label="QF"):
    """Draw one team node: a circle with its code, name, and a probability."""
    ax.add_patch(Circle((x, y), r, facecolor=NODE_FACE, edgecolor=NODE_EDGE,
                         lw=2.5, zorder=3))
    ax.text(x, y + 0.02, _code(team), ha="center", va="center",
            fontsize=9, fontweight="bold", color=TEXT_DARK, zorder=4)
    ax.text(x, y - r - 0.16, team, ha="center", va="center",
            fontsize=7.5, color="white", zorder=4)
    ax.text(x, y - r - 0.34, f"{prob_label} {prob*100:.0f}%", ha="center",
            va="center", fontsize=6.5, color=ACCENT, zorder=4)


def _elbow(ax, x0, y0, x1, y1):
    """Draw a bracket-style connector from (x0,y0) toward (x1,y1)."""
    xm = (x0 + x1) / 2
    ax.plot([x0, xm], [y0, y0], color=LINE, lw=1.6, zorder=1)
    ax.plot([xm, xm], [y0, y1], color=LINE, lw=1.6, zorder=1)
    ax.plot([xm, x1], [y1, y1], color=LINE, lw=1.6, zorder=1)


def plot_quarter_final_bracket(team_stage: pd.DataFrame,
                               path: Path | None = None) -> Path:
    """Render the most-likely QF bracket poster from the team-stage table."""
    path = path or (config.CHARTS_DIR / "quarter_final_bracket.png")

    ts = team_stage.copy()
    qf8 = ts.nlargest(8, "prob_reach_qf").reset_index(drop=True)
    # Balance the bracket so strong teams meet later: seed order 1,4,5,8 left;
    # 2,3,6,7 right (a standard seeding split).
    left_idx = [0, 3, 4, 7]
    right_idx = [1, 2, 5, 6]
    left = qf8.iloc[left_idx].reset_index(drop=True)
    right = qf8.iloc[right_idx].reset_index(drop=True)

    # Each side's finalist is the strongest of that side's four teams, so the
    # drawn bracket is internally consistent. The champion is the finalist with
    # the higher overall title probability.
    fin_left = left.nlargest(1, "prob_reach_final").iloc[0]
    fin_right = right.nlargest(1, "prob_reach_final").iloc[0]
    champ = fin_left if fin_left["prob_champion"] >= fin_right["prob_champion"] else fin_right

    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Subtle halftone-dot texture, fading left-to-right like the reference.
    rng = np.random.default_rng(2026)
    dots = rng.uniform([0, 0], [10, 6.2], size=(600, 2))
    sizes = (10 - dots[:, 0]) * 1.2 + 1
    ax.scatter(dots[:, 0], dots[:, 1], s=sizes, color="#9c2c58", alpha=0.5, zorder=0)

    # Y positions for the 4 QF nodes on each side.
    qf_ys = [5.4, 4.0, 2.2, 0.8]
    sf_ys = [4.7, 1.5]
    fin_y = 3.1

    lx_qf, lx_sf, lx_fin = 0.7, 2.6, 4.2
    rx_qf, rx_sf, rx_fin = 9.3, 7.4, 5.8
    cx = 5.0

    # Left side nodes + connectors.
    for i, y in enumerate(qf_ys):
        row = left.iloc[i]
        _draw_node(ax, lx_qf, y, row["team"], row["prob_reach_qf"], prob_label="QF")
    for k, sy in enumerate(sf_ys):
        _elbow(ax, lx_qf + 0.42, qf_ys[2 * k], lx_sf, sy)
        _elbow(ax, lx_qf + 0.42, qf_ys[2 * k + 1], lx_sf, sy)
    # SF nodes (left) labelled with the better of the two feeding teams' SF prob.
    for k, sy in enumerate(sf_ys):
        feed = left.iloc[[2 * k, 2 * k + 1]].nlargest(1, "prob_reach_sf").iloc[0]
        _draw_node(ax, lx_sf, sy, feed["team"], feed["prob_reach_sf"], prob_label="SF")
    # Finalist node (left).
    _elbow(ax, lx_sf + 0.42, sf_ys[0], lx_fin, fin_y)
    _elbow(ax, lx_sf + 0.42, sf_ys[1], lx_fin, fin_y)
    _draw_node(ax, lx_fin, fin_y, fin_left["team"], fin_left["prob_reach_final"],
               prob_label="Final")

    # Right side nodes + connectors (mirrored).
    for i, y in enumerate(qf_ys):
        row = right.iloc[i]
        _draw_node(ax, rx_qf, y, row["team"], row["prob_reach_qf"], prob_label="QF")
    for k, sy in enumerate(sf_ys):
        _elbow(ax, rx_qf - 0.42, qf_ys[2 * k], rx_sf, sy)
        _elbow(ax, rx_qf - 0.42, qf_ys[2 * k + 1], rx_sf, sy)
    for k, sy in enumerate(sf_ys):
        feed = right.iloc[[2 * k, 2 * k + 1]].nlargest(1, "prob_reach_sf").iloc[0]
        _draw_node(ax, rx_sf, sy, feed["team"], feed["prob_reach_sf"], prob_label="SF")
    _elbow(ax, rx_sf - 0.42, sf_ys[0], rx_fin, fin_y)
    _elbow(ax, rx_sf - 0.42, sf_ys[1], rx_fin, fin_y)
    _draw_node(ax, rx_fin, fin_y, fin_right["team"], fin_right["prob_reach_final"],
               prob_label="Final")

    # Centre FINAL connector + projected champion label (in the gap).
    ax.plot([lx_fin + 0.42, rx_fin - 0.42], [fin_y, fin_y], color=LINE, lw=1.6, zorder=1)
    ax.text(cx, fin_y + 0.42, "FINAL", ha="center", va="center", fontsize=15,
            fontweight="bold", color="white", zorder=5)
    ax.text(cx, fin_y - 0.30, _code(champ["team"]), ha="center", va="center",
            fontsize=10, fontweight="bold", color="white", zorder=5)
    ax.text(cx, fin_y - 0.52, f"Champion {champ['prob_champion']*100:.0f}%",
            ha="center", va="center", fontsize=7, color=ACCENT, zorder=5)

    # Titles.
    ax.text(cx, 6.3, "QUARTER FINAL", ha="center", va="center", fontsize=20,
            fontweight="bold", color="white")
    ax.text(cx, 5.92, "2026 WORLD CUP — PROJECTED BRACKET", ha="center",
            va="center", fontsize=9.5, color=ACCENT, fontweight="bold")

    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(0.0, 6.6)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)
    return path


def generate_bracket(tables_dir: Path | None = None) -> Path:
    """Convenience: load the team-stage table and render the bracket."""
    config.ensure_dirs()
    tables_dir = tables_dir or config.TABLES_DIR
    ts = pd.read_csv(tables_dir / "team_stage_probabilities.csv")
    return plot_quarter_final_bracket(ts)

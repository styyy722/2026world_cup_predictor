"""Projected quarter-final bracket chart (minimalist style).

Builds a "most likely" knockout bracket from the aggregated simulation output
(`team_stage_probabilities.csv`): the 8 teams most likely to reach the
quarter-finals are placed at the QF slots, each side's strongest team becomes
its finalist, and the higher title probability of the two is the projected
champion. Every node is annotated with the relevant stage probability.

Design is intentionally minimal: light background, thin neutral connectors,
small dot markers, and a single accent colour reserved for the champion.
Layout: QF (4 per side) -> SF (2 per side) -> finalist (1 per side) -> FINAL.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .. import config

# Minimalist palette: off-white canvas, soft grey lines, near-black text,
# one restrained accent for the champion.
BG = "#fbfbf9"
LINE = "#c8c8c8"
TEXT = "#222222"
MUTED = "#9a9a9a"
ACCENT = "#c8a24a"  # muted gold, used only for the projected champion

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


def _draw_node(ax, x, y, team, prob, prob_label, side="left",
               accent=False, big=False):
    """Draw a minimal node: a small dot plus left/right-aligned label text.

    Text is placed on the open side so it never overlaps the connectors.
    """
    color = ACCENT if accent else TEXT
    ax.plot([x], [y], marker="o", markersize=6 if not big else 8,
            markerfacecolor=color, markeredgecolor=color, zorder=3)

    name_size = 11 if big else 9.5
    if side == "top":
        # Label stacked vertically, centred on the dot (used at the centre,
        # where horizontal labels would clash with the FINAL block).
        ax.text(x, y + 0.24, team, ha="center", va="center", fontsize=name_size,
                fontweight="bold", color=color, zorder=4)
        ax.text(x, y - 0.24, f"{prob_label} {prob*100:.0f}%", ha="center",
                va="center", fontsize=7.5, color=MUTED, zorder=4)
        return

    if side == "left":
        tx, ha = x + 0.18, "left"
    else:
        tx, ha = x - 0.18, "right"
    ax.text(tx, y + 0.16, team, ha=ha, va="center", fontsize=name_size,
            fontweight="bold", color=color, zorder=4)
    ax.text(tx, y - 0.16, f"{prob_label} {prob*100:.0f}%", ha=ha, va="center",
            fontsize=7.5, color=MUTED, zorder=4)


def _elbow(ax, x0, y0, x1, y1):
    """Thin bracket-style connector from (x0,y0) toward (x1,y1)."""
    xm = (x0 + x1) / 2
    ax.plot([x0, xm], [y0, y0], color=LINE, lw=1.0, zorder=1)
    ax.plot([xm, xm], [y0, y1], color=LINE, lw=1.0, zorder=1)
    ax.plot([xm, x1], [y1, y1], color=LINE, lw=1.0, zorder=1)


def plot_quarter_final_bracket(team_stage: pd.DataFrame,
                               path: Path | None = None) -> Path:
    """Render the most-likely QF bracket from the team-stage table."""
    path = path or (config.CHARTS_DIR / "quarter_final_bracket.png")

    ts = team_stage.copy()
    qf8 = ts.nlargest(8, "prob_reach_qf").reset_index(drop=True)
    # Balance the bracket so the strongest teams meet later.
    left = qf8.iloc[[0, 3, 4, 7]].reset_index(drop=True)
    right = qf8.iloc[[1, 2, 5, 6]].reset_index(drop=True)

    fin_left = left.nlargest(1, "prob_reach_final").iloc[0]
    fin_right = right.nlargest(1, "prob_reach_final").iloc[0]
    champ = fin_left if fin_left["prob_champion"] >= fin_right["prob_champion"] else fin_right

    fig, ax = plt.subplots(figsize=(11, 6.0))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    qf_ys = [5.3, 4.1, 2.1, 0.9]
    sf_ys = [4.7, 1.5]
    fin_y = 3.1

    lx_qf, lx_sf, lx_fin = 0.6, 2.9, 4.0
    rx_qf, rx_sf, rx_fin = 9.4, 7.1, 6.0
    cx = 5.0

    # ---- Left side ----
    for i, y in enumerate(qf_ys):
        r = left.iloc[i]
        _draw_node(ax, lx_qf, y, r["team"], r["prob_reach_qf"], "QF", side="left")
    for k, sy in enumerate(sf_ys):
        _elbow(ax, lx_qf, qf_ys[2 * k], lx_sf, sy)
        _elbow(ax, lx_qf, qf_ys[2 * k + 1], lx_sf, sy)
        feed = left.iloc[[2 * k, 2 * k + 1]].nlargest(1, "prob_reach_sf").iloc[0]
        _draw_node(ax, lx_sf, sy, feed["team"], feed["prob_reach_sf"], "SF", side="left")
    _elbow(ax, lx_sf, sf_ys[0], lx_fin, fin_y)
    _elbow(ax, lx_sf, sf_ys[1], lx_fin, fin_y)
    _draw_node(ax, lx_fin, fin_y, fin_left["team"], fin_left["prob_reach_final"],
               "Final", side="top", accent=(champ["team"] == fin_left["team"]))

    # ---- Right side ----
    for i, y in enumerate(qf_ys):
        r = right.iloc[i]
        _draw_node(ax, rx_qf, y, r["team"], r["prob_reach_qf"], "QF", side="right")
    for k, sy in enumerate(sf_ys):
        _elbow(ax, rx_qf, qf_ys[2 * k], rx_sf, sy)
        _elbow(ax, rx_qf, qf_ys[2 * k + 1], rx_sf, sy)
        feed = right.iloc[[2 * k, 2 * k + 1]].nlargest(1, "prob_reach_sf").iloc[0]
        _draw_node(ax, rx_sf, sy, feed["team"], feed["prob_reach_sf"], "SF", side="right")
    _elbow(ax, rx_sf, sf_ys[0], rx_fin, fin_y)
    _elbow(ax, rx_sf, sf_ys[1], rx_fin, fin_y)
    _draw_node(ax, rx_fin, fin_y, fin_right["team"], fin_right["prob_reach_final"],
               "Final", side="top", accent=(champ["team"] == fin_right["team"]))

    # ---- Centre: FINAL + projected champion, stacked below the line ----
    ax.plot([lx_fin, rx_fin], [fin_y, fin_y], color=LINE, lw=1.0, zorder=1)
    ax.text(cx, fin_y - 0.55, "FINAL", ha="center", va="center", fontsize=11,
            fontweight="bold", color=MUTED, zorder=5)
    ax.text(cx, fin_y - 0.90, champ["team"], ha="center", va="center",
            fontsize=13, fontweight="bold", color=ACCENT, zorder=5)
    ax.text(cx, fin_y - 1.18, f"Champion · {champ['prob_champion']*100:.0f}%",
            ha="center", va="center", fontsize=8.5, color=MUTED, zorder=5)

    # ---- Title ----
    ax.text(cx, 5.95, "2026 World Cup — Projected Bracket", ha="center",
            va="center", fontsize=13, fontweight="bold", color=TEXT)

    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(0.2, 6.3)
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

"""Streamlit dashboard for forecast outputs.

Run with:
    streamlit run dashboard.py
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config


st.set_page_config(
    page_title="2026 World Cup Forecast",
    layout="wide",
)


@st.cache_data
def _load_csv(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _pct(series: pd.Series) -> pd.Series:
    return (series * 100).round(1)


team_stage = _load_csv(config.TABLES_DIR / "team_stage_probabilities.csv")
match_probs = _load_csv(config.TABLES_DIR / "match_probabilities.csv")
summary = _load_csv(config.TABLES_DIR / "simulation_summary.csv")
backtest = _load_csv(config.TABLES_DIR / "backtest_summary.csv")

st.title("2026 World Cup Forecast")

if team_stage.empty:
    st.warning(
        "No simulation output found. Run "
        "`python main.py --mode simulate --model elo` or a trained model first."
    )
    st.stop()

top = team_stage.nlargest(8, "prob_champion")
cols = st.columns(4)
cols[0].metric("Teams", f"{len(team_stage):,}")
cols[1].metric("Top contender", top.iloc[0]["team"])
cols[2].metric("Champion odds", f"{top.iloc[0]['prob_champion'] * 100:.1f}%")
cols[3].metric("Simulations", f"{len(summary):,}" if not summary.empty else "n/a")

tab1, tab2, tab3, tab4 = st.tabs([
    "Title Race",
    "Groups",
    "Matches",
    "Backtests",
])

with tab1:
    plot_df = top.assign(prob_champion_pct=_pct(top["prob_champion"]))
    plot_df = plot_df.sort_values("prob_champion_pct")
    fig = go.Figure(
        go.Bar(
            x=plot_df["prob_champion_pct"],
            y=plot_df["team"],
            orientation="h",
            marker_color="#2563eb",
            text=plot_df["group"].map(lambda group: f"Group {group}"),
            hovertemplate=(
                "%{y}<br>Champion probability: %{x:.1f}%<br>%{text}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=440,
        xaxis_title="Champion probability (%)",
        yaxis_title="",
        margin={"l": 20, "r": 20, "t": 20, "b": 40},
    )
    st.plotly_chart(fig, width="stretch")

    display = team_stage.copy()
    for col in [c for c in display.columns if c.startswith("prob_")]:
        display[col] = _pct(display[col])
    st.dataframe(display, width="stretch", hide_index=True)

with tab2:
    group = st.selectbox("Group", sorted(team_stage["group"].dropna().unique()))
    group_df = team_stage[team_stage["group"] == group].copy()
    group_df["expected_group_points"] = group_df["expected_group_points"].round(2)
    for col in [
        "prob_group_1st",
        "prob_group_2nd",
        "prob_group_3rd",
        "prob_reach_r32",
        "prob_champion",
    ]:
        group_df[col] = _pct(group_df[col])
    st.dataframe(
        group_df[[
            "team",
            "expected_group_points",
            "prob_group_1st",
            "prob_group_2nd",
            "prob_group_3rd",
            "prob_reach_r32",
            "prob_champion",
        ]],
        width="stretch",
        hide_index=True,
    )

with tab3:
    if match_probs.empty:
        st.info("No match probability table found yet.")
    else:
        st.dataframe(match_probs, width="stretch", hide_index=True)

with tab4:
    if backtest.empty:
        st.info("No backtest summary found yet.")
    else:
        st.dataframe(backtest, width="stretch", hide_index=True)

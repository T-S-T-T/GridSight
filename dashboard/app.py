"""GridSight dashboard. Run with ``streamlit run dashboard/app.py``.

Reads only stored outputs (predictions, reports, silver tables); it never trains or scores.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gridsight import REGIONS, storage
from gridsight.analysis import market
from gridsight.config import get_settings
from gridsight.serving import SERVING_LAYER
from gridsight.transform.quality import freshness

st.set_page_config(page_title="GridSight", layout="wide")
S = get_settings()


@st.cache_data(ttl=60)
def load(table: str, layer: str = "silver") -> pd.DataFrame:
    return storage.read(table, layer=layer)


@st.cache_data(ttl=60)
def load_csv(rel: str) -> pd.DataFrame:
    path = S.reports_dir / rel
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:  # e.g. no AEMO comparison exists for the spike task
        return pd.DataFrame()


def fan_chart(pred: pd.DataFrame, title: str, unit: str) -> go.Figure:
    fig = go.Figure()
    model = pred[~pred["model"].str.endswith("aemo_predispatch")].sort_values("target_utc")
    aemo = pred[pred["model"].str.endswith("aemo_predispatch")].sort_values("target_utc")
    if {"q0.1", "q0.9"} <= set(model.columns) and model["q0.1"].notna().any():
        fig.add_trace(go.Scatter(x=model["target_utc"], y=model["q0.9"], line={"width": 0}, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=model["target_utc"], y=model["q0.1"], fill="tonexty", line={"width": 0}, name="GridSight 80% interval"))
    fig.add_trace(go.Scatter(x=model["target_utc"], y=model["q0.5"], name="GridSight p50", line={"width": 2.5}))
    if not aemo.empty:
        fig.add_trace(go.Scatter(x=aemo["target_utc"], y=aemo["q0.5"], name="AEMO pre-dispatch", line={"dash": "dash"}))
    fig.update_layout(title=title, yaxis_title=unit, height=360, margin={"t": 50, "b": 30})
    return fig


st.title("GridSight")
st.caption("Forecasts for Australia's National Electricity Market, benchmarked against AEMO pre-dispatch.")

latest = storage.latest_time("dispatch_price")
fr = freshness(latest)
if latest is None:
    st.warning("No data yet. Run `gridsight demo` (offline) or `gridsight backfill` first.")
    st.stop()
if fr.stale:
    st.warning(f"Data is stale: latest interval {latest:%Y-%m-%d %H:%M} UTC.")

page = st.sidebar.radio("Page", ["Now", "Accuracy", "Battery", "Explain", "Market story"])
region = st.sidebar.selectbox("Region", REGIONS, index=REGIONS.index("SA1"))

if page == "Now":
    price = load("dispatch_price")
    now = price[price["interval_utc"] == price["interval_utc"].max()].set_index("region")["rrp"]
    cols = st.columns(len(REGIONS))
    for c, r in zip(cols, REGIONS, strict=True):
        c.metric(r, f"${now.get(r, float('nan')):,.0f}/MWh")
    pred = load("predictions", SERVING_LAYER)
    if pred.empty:
        st.info("No live predictions yet: run `gridsight score`.")
    else:
        latest_pred = pred[(pred["region"] == region) & (pred["issued_utc"] == pred["issued_utc"].max())]
        spike = latest_pred[latest_pred["task"] == "spike"]
        if not spike.empty:
            p = float(spike["p_spike"].max())
            (st.error if p >= 0.5 else st.warning if p >= 0.2 else st.success)(f"Spike risk in the next hour: {p:.0%}")
        for task, unit in (("price", "$/MWh"), ("demand", "MW")):
            t = latest_pred[latest_pred["task"] == task]
            if not t.empty:
                st.plotly_chart(fan_chart(t, f"{region} {task} forecast", unit), use_container_width=True)

elif page == "Accuracy":
    st.subheader("Locked test period")
    for task in ("demand", "price", "spike"):
        test = load_csv(f"results/{task}_test.csv")
        if test.empty:
            continue
        st.markdown(f"**{task.title()}**")
        st.dataframe(test, hide_index=True)
        vs = load_csv(f"results/{task}_vs_aemo_by_horizon.csv")
        if not vs.empty:
            metric = [c for c in vs.columns if c.startswith("model_")][0]
            base = metric.replace("model_", "aemo_")
            fig = go.Figure()
            for m, g in vs.groupby("model"):
                fig.add_trace(go.Scatter(x=g["horizon"] / 2, y=g[metric], name=m, mode="lines+markers"))
            first = vs[vs["model"] == vs["model"].iloc[0]]
            fig.add_trace(go.Scatter(x=first["horizon"] / 2, y=first[base], name="AEMO pre-dispatch", line={"dash": "dash"}))
            fig.update_layout(xaxis_title="horizon (hours)", yaxis_title=metric.replace("model_", "").upper(), height=320)
            st.plotly_chart(fig, use_container_width=True)
    if load_csv("results/demand_test.csv").empty:
        st.info("Run `gridsight evaluate` to produce results.")

elif page == "Battery":
    summary = load_csv("battery/summary.csv")
    if summary.empty:
        st.info("Run `gridsight battery` first.")
    else:
        st.dataframe(summary.round(1), hide_index=True)
        runs_path = S.reports_dir / "battery" / "runs.parquet"
        if runs_path.exists():
            runs = pd.read_parquet(runs_path)
            fig = go.Figure()
            for name, g in runs.groupby("strategy"):
                fig.add_trace(go.Scatter(x=g["interval_utc"], y=g["revenue_aud"].cumsum(), name=name))
            fig.update_layout(title="Cumulative revenue", yaxis_title="AUD", height=380)
            st.plotly_chart(fig, use_container_width=True)

elif page == "Explain":
    from gridsight.analysis.explain import explain_spike, global_importance
    from gridsight.evaluate.run import load_features
    from gridsight.models.base import load_model

    try:
        model, card = load_model("spike_lgbm")
    except FileNotFoundError:
        st.info("No Production spike model yet: run `gridsight train`.")
        st.stop()
    feats = load_features()
    f = feats[(feats["region"] == region) & (feats["horizon"] <= 2)]
    if f.empty:
        st.info("No features for this region.")
    else:
        st.subheader(f"What drives spike risk ({card.version})")
        imp = global_importance(model, f.tail(3000), top=10)
        st.bar_chart(imp.set_index("label")["mean_abs_shap"])
        latest_row = f[f["issued_utc"] == f["issued_utc"].max()].head(1)
        st.write(explain_spike(model, latest_row))

else:  # Market story
    price, demand = load("dispatch_price"), load("region_demand")
    st.dataframe(market.market_summary(price, demand, S.spike_threshold).round(2), hide_index=True)
    curve = market.duck_curve(demand[demand["region"] == region], by="month")
    fig = go.Figure()
    for m, g in curve.groupby("month"):
        fig.add_trace(go.Scatter(x=g["hour"], y=g["total_demand_mw"], name=f"month {m}"))
    fig.update_layout(title=f"{region} average demand by time of day", xaxis_title="hour (market time)", yaxis_title="MW")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(market.spike_stats(price, S.spike_threshold).round(3), hide_index=True)

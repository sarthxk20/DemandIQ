import logging
import warnings

# ── Suppress noisy Prophet / Stan output before any imports ──────────────────
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from statsmodels.tsa.seasonal import STL

from src.data_loader import load_data
from src.prophet_model import prophet_forecast
from src.insight_engine import generate_business_insight

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DemandIQ — Retail Demand Forecasting",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("<h1 style='font-size:64px; margin-bottom:0;'>DemandIQ</h1>",
            unsafe_allow_html=True)
st.subheader("Retail Demand Forecasting & Risk Insights")
st.markdown(
    "This dashboard explains **how demand behaves**, **what to expect next**, "
    "and **how to make better inventory decisions**."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD & VALIDATE DATA
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_cached_data():
    return load_data()

train = load_cached_data()

required_columns = {"Store", "Date", "Sales"}
if not required_columns.issubset(train.columns):
    st.error(
        "Dataset schema mismatch. This app expects columns: **Store**, **Date**, **Sales**.\n\n"
        "Please provide a dataset with these columns."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.header("Store & Forecast Settings")

store_ids = sorted(train["Store"].unique())
STORE_ID  = st.sidebar.selectbox("Select Store ID", store_ids)

HORIZON = st.sidebar.select_slider(
    "Forecast Horizon (days)",
    options=[7, 14, 21, 30],
    value=14,
    help="How many days ahead to forecast. Longer horizons carry more uncertainty.",
)

st.sidebar.divider()

# Date range filter
date_min_global = train["Date"].min()
date_max_global = train["Date"].max()

date_range = st.sidebar.date_input(
    "Historical date range to display",
    value=(date_min_global, date_max_global),
    min_value=date_min_global,
    max_value=date_max_global,
    help="Zoom into a specific period. The forecast always uses the full history.",
)
# Unpack safely — user may select only one date
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    display_start, display_end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    display_start = pd.Timestamp(date_range[0]) if date_range else date_min_global
    display_end   = date_max_global

st.sidebar.divider()
st.sidebar.caption("DemandIQ · Retail Forecasting · Prophet + Streamlit")

# ─────────────────────────────────────────────────────────────────────────────
# BUILD STORE SERIES (always full history for model, filtered for display)
# ─────────────────────────────────────────────────────────────────────────────

store_df = train[train["Store"] == STORE_ID].copy().sort_values("Date")
store_df.set_index("Date", inplace=True)

series_full    = store_df["Sales"].asfreq("D", fill_value=0)   # full — for model
series_display = series_full[display_start:display_end]         # filtered — for charts

# ─────────────────────────────────────────────────────────────────────────────
# CACHED FORECAST  (keyed on store, horizon — recomputes only when they change)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_forecast(store_id: int, horizon: int) -> pd.DataFrame:
    """Run Prophet on the full store series. Cached by store + horizon."""
    _store_df = train[train["Store"] == store_id].copy().sort_values("Date")
    _store_df.set_index("Date", inplace=True)
    _series   = _store_df["Sales"].asfreq("D", fill_value=0)
    return prophet_forecast(_series, horizon=horizon, return_intervals=True)


@st.cache_data(show_spinner=False)
def get_backtest(store_id: int) -> dict:
    """
    Rolling 80/20 backtest: train on first 80% of store data,
    predict the last 20%, return actuals + predictions.
    """
    _store_df = train[train["Store"] == store_id].copy().sort_values("Date")
    _store_df.set_index("Date", inplace=True)
    _series   = _store_df["Sales"].asfreq("D", fill_value=0)

    cutoff    = int(len(_series) * 0.8)
    if cutoff < 30:
        return {}   # not enough data

    train_s   = _series.iloc[:cutoff]
    test_s    = _series.iloc[cutoff:]
    horizon_bt= len(test_s)

    bt_df     = prophet_forecast(train_s, horizon=horizon_bt, return_intervals=True)
    bt_df     = bt_df.iloc[:horizon_bt]

    mae  = float(np.mean(np.abs(test_s.values - bt_df["yhat"].values)))
    mape = float(np.mean(np.abs((test_s.values - bt_df["yhat"].values) /
                                 np.where(test_s.values == 0, 1, test_s.values)))) * 100

    return {
        "actuals":     test_s,
        "predictions": bt_df["yhat"],
        "lower":       bt_df["yhat_lower"],
        "upper":       bt_df["yhat_upper"],
        "mae":         mae,
        "mape":        mape,
        "cutoff_date": train_s.index[-1],
    }

# ─────────────────────────────────────────────────────────────────────────────
# STL DECOMPOSITION (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_stl(store_id: int):
    _store_df = train[train["Store"] == store_id].copy().sort_values("Date")
    _store_df.set_index("Date", inplace=True)
    _series   = _store_df["Sales"].asfreq("D", fill_value=0)
    stl       = STL(_series, period=7, robust=True)
    return stl.fit(), _series

# ─────────────────────────────────────────────────────────────────────────────
# TABS  (replaces sidebar anchor links — actually navigable)
# ─────────────────────────────────────────────────────────────────────────────

(tab_summary, tab_quality, tab_behavior, tab_changed,
 tab_drivers, tab_models, tab_forecast, tab_inventory,
 tab_scenario, tab_anomaly, tab_insight) = st.tabs([
    "Executive Summary",
    "Data Quality",
    "Sales Behavior",
    "What Changed",
    "Demand Drivers",
    "Model Comparison",
    "Forecast & Risk",
    "Inventory",
    "Scenario Simulation",
    "Anomaly Detection",
    "Final Insight",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB: EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab_summary:
    st.header("Executive Summary")
    st.markdown(
        """
        **In simple terms:**  
        Sales follow a **weekly rhythm** — some days consistently perform better than others.

        **Why this matters:**  
        Understanding this pattern allows teams to **plan ahead** instead of reacting late.

        **Impact:**  
        Seasonality-aware forecasting reduces errors by approximately **60%**, helping prevent
        stockouts and unnecessary overstocking.
        """
    )

    # Quick KPIs
    if not series_full.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total history",     f"{len(series_full):,} days")
        c2.metric("Avg daily sales",   f"{series_full.mean():,.0f} units")
        c3.metric("Peak day",          f"{series_full.max():,.0f} units",
                  delta=f"{series_full.idxmax().strftime('%d %b %Y')}")
        c4.metric("Forecast horizon",  f"{HORIZON} days")

# ══════════════════════════════════════════════════════════════════════════════
# TAB: DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════
with tab_quality:
    st.header("Data Quality Snapshot")

    if store_df.empty:
        st.warning("No data available for this store.")
    else:
        date_min     = store_df.index.min()
        date_max     = store_df.index.max()
        expected     = pd.date_range(start=date_min, end=date_max, freq="D")
        missing_count= len(expected.difference(store_df.index))
        zero_days    = int((store_df["Sales"] == 0).sum())

        sales_values = store_df["Sales"]
        q1, q3       = sales_values.quantile(0.25), sales_values.quantile(0.75)
        iqr          = q3 - q1
        if iqr == 0:
            outlier_count = 0
        else:
            lower         = q1 - 1.5 * iqr
            upper         = q3 + 1.5 * iqr
            outlier_count = int(((sales_values < lower) | (sales_values > upper)).sum())

        col_missing, col_zero, col_outlier = st.columns(3)
        col_missing.metric("Missing dates",     missing_count)
        col_zero.metric("Zero-sales days",      zero_days)
        col_outlier.metric("Outlier days (IQR)",outlier_count)

        st.caption(
            "Missing dates = gaps before daily forward-fill. "
            "Outliers use the IQR ×1.5 fence rule."
        )

        # Small data completeness bar
        total_expected = len(expected)
        completeness   = 100 * (1 - missing_count / max(total_expected, 1))
        st.progress(int(completeness), text=f"Data completeness: {completeness:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# TAB: SALES BEHAVIOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_behavior:
    st.header("How has this store been selling historically?")
    st.markdown(
        """
        **What this chart shows:** Daily sales over the selected date range.  
        **What to notice:** Repeating ups and downs — predictable weekly patterns.  
        **Why this matters:** Predictable demand enables reliable forecasting.
        """
    )

    fig_sales = px.line(
        series_display,
        title=f"Daily Sales — Store {STORE_ID}  "
              f"({display_start.strftime('%d %b %Y')} → {display_end.strftime('%d %b %Y')})",
        labels={"value": "Units Sold", "index": "Date"},
    )
    fig_sales.update_traces(line_color="#1f77b4")
    st.plotly_chart(fig_sales, width="stretch")
    st.caption(
        "Use the **Historical date range** slider in the sidebar to zoom in. "
        "The forecast always trains on the full history regardless of this filter."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB: WHAT CHANGED
# ══════════════════════════════════════════════════════════════════════════════
with tab_changed:
    st.header("What changed recently?")
    st.markdown(
        "Comparing the most recent period against the equivalent prior period "
        "to surface meaningful shifts in demand."
    )

    if len(series_full) < 14:
        st.info("Not enough history (need at least 14 days) to summarise recent changes.")
    else:
        last_7   = series_full[-7:].mean()
        prev_7   = series_full[-14:-7].mean()
        delta_7  = last_7 - prev_7
        pct_7    = delta_7 / prev_7 if prev_7 != 0 else 0.0

        # 30-day comparison
        if len(series_full) >= 60:
            last_30  = series_full[-30:].mean()
            prev_30  = series_full[-60:-30].mean()
            delta_30 = last_30 - prev_30
            pct_30   = delta_30 / prev_30 if prev_30 != 0 else 0.0
            have_30  = True
        else:
            have_30  = False

        # Metric cards with delta arrows
        cols = st.columns(4 if have_30 else 2)

        cols[0].metric(
            "Last 7 days (avg / day)",
            f"{last_7:,.0f}",
            delta=f"{pct_7*100:+.1f}% vs prior 7d",
            delta_color="normal",
            help="Average daily sales in the last 7 days vs the 7 days before that.",
        )
        cols[1].metric(
            "Prior 7 days (avg / day)",
            f"{prev_7:,.0f}",
            help="Baseline period for the 7-day comparison.",
        )
        if have_30:
            cols[2].metric(
                "Last 30 days (avg / day)",
                f"{last_30:,.0f}",
                delta=f"{pct_30*100:+.1f}% vs prior 30d",
                delta_color="normal",
            )
            cols[3].metric(
                "Prior 30 days (avg / day)",
                f"{prev_30:,.0f}",
            )

        # Mini trend bar: last 14 days
        st.markdown("**Last 14 days — daily sales**")
        last14 = series_full[-14:].reset_index()
        last14.columns = ["Date", "Sales"]
        fig_14 = px.bar(
            last14, x="Date", y="Sales",
            color="Sales", color_continuous_scale="Blues",
            title="Recent 14-Day Sales Bar",
        )
        fig_14.update_layout(coloraxis_showscale=False, height=300)
        st.plotly_chart(fig_14, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB: DEMAND DRIVERS
# ══════════════════════════════════════════════════════════════════════════════
with tab_drivers:
    st.header("What is driving sales changes?")
    st.markdown(
        """
        Sales are decomposed into three interpretable components:

        - **Trend** — Long-term growth or decline  
        - **Weekly Pattern (Seasonality)** — Regular recurring behavior by day of week  
        - **Residual (Unexpected Changes)** — Promotions, disruptions, or data anomalies  

        **Why this matters:** Understanding these drivers determines the right forecasting strategy.
        """
    )

    with st.spinner("Decomposing sales signal..."):
        stl_result, stl_series = get_stl(STORE_ID)

    decomp_df = pd.DataFrame({
        "Observed Sales":               stl_result.observed,
        "Trend (Long-Term Movement)":   stl_result.trend,
        "Weekly Pattern (Seasonality)": stl_result.seasonal,
        "Residual (Unexpected Changes)":stl_result.resid,
    })
    # Apply display date filter
    decomp_df = decomp_df[display_start:display_end]

    fig_decomp = px.line(
        decomp_df,
        facet_row="variable",
        height=800,
        title="Breaking Down Sales Behavior",
    )
    fig_decomp.update_yaxes(title_text="")
    fig_decomp.update_layout(legend_title_text="", margin=dict(l=200, r=40))
    for ann in fig_decomp.layout.annotations:
        if ann.text.startswith("variable="):
            ann.update(
                text=ann.text.split("=", 1)[1],
                textangle=0, x=0, xanchor="right", yanchor="middle",
            )
    st.plotly_chart(fig_decomp, width="stretch")

    st.info(
        "**Key takeaway:** Weekly seasonality explains most of the sales variation. "
        "Residuals highlight unusual or one-off events."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab_models:
    st.header("Which forecasting approach works best?")
    st.markdown(
        """
        **How to read this chart:**  
        Each bar is a forecasting approach. **Lower bars = better accuracy** (lower average error).

        **Why this matters:** More accurate forecasts directly reduce inventory risk.
        """
    )

    comparison_df = pd.DataFrame({
        "Model":              ["Naive", "Moving Average", "ARIMA", "SARIMA", "Prophet"],
        "Average Error (MAE)":[1993.86,  1519.18,          1412.63,  784.21,   778.15],
    })

    fig_comp = px.bar(
        comparison_df, x="Model", y="Average Error (MAE)",
        text_auto=True,
        color="Average Error (MAE)",
        color_continuous_scale="RdYlGn_r",
        title="Forecast Accuracy Comparison (Lower is Better)",
    )
    fig_comp.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_comp, width="stretch")

    st.caption(
        "⚠️ These MAE values are **averaged across all stores** in the dataset and shown "
        "for illustrative comparison only. For store-specific accuracy, see the "
        "**Forecast & Risk** tab which runs a live backtest on the selected store."
    )

    st.success(
        "Seasonality-aware models (SARIMA, Prophet) reduce forecasting errors by ~**60%** "
        "compared to a naive baseline."
    )

    st.markdown("**Why Prophet over SARIMA?**")
    st.markdown(
        """
        Prophet performs on par with SARIMA while being:
        - Easier to maintain and retrain  
        - More robust to missing data and shifting patterns  
        - Faster to deploy in production  

        This makes it better suited for **real-world business use**.
        """
    )

    # ── Live backtest for selected store ──────────────────────────────────────
    st.subheader(f"Live Backtest — Store {STORE_ID}")
    st.markdown(
        "Prophet is trained on the **first 80%** of this store's history, "
        "then tested on the **last 20%** it never saw. "
        "This shows how the model would have performed in practice."
    )

    with st.spinner("Running backtest (training on 80%, testing on 20%)..."):
        bt = get_backtest(STORE_ID)

    if not bt:
        st.warning("Not enough data to run a backtest for this store (need at least 38 days).")
    else:
        b1, b2 = st.columns(2)
        b1.metric(
            "Backtest MAE",
            f"{bt['mae']:,.0f} units",
            help="Mean Absolute Error on the held-out 20% — lower is better.",
        )
        b2.metric(
            "Backtest MAPE",
            f"{bt['mape']:.1f}%",
            help="Mean Absolute Percentage Error — how wrong the model is on average, as a %.",
        )

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=bt["actuals"].index, y=bt["actuals"].values,
            mode="lines", name="Actual Sales", line=dict(color="#1f77b4"),
        ))
        fig_bt.add_trace(go.Scatter(
            x=bt["predictions"].index, y=bt["predictions"].values,
            mode="lines", name="Prophet Prediction",
            line=dict(color="#ff7f0e", dash="dash"),
        ))
        fig_bt.add_trace(go.Scatter(
            x=bt["upper"].index, y=bt["upper"].values,
            mode="lines", line=dict(width=0), showlegend=False,
        ))
        fig_bt.add_trace(go.Scatter(
            x=bt["lower"].index, y=bt["lower"].values,
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(255, 127, 14, 0.15)",
            name="Prediction Interval",
        ))
        # add_vline doesn't handle datetime strings reliably across Plotly versions;
        # a vertical scatter trace is the safe cross-version alternative.
        fig_bt.add_trace(go.Scatter(
            x=[bt["cutoff_date"], bt["cutoff_date"]],
            y=[0, float(bt["actuals"].max()) * 1.1],
            mode="lines",
            line=dict(color="grey", dash="dot", width=1.5),
            name="Train / Test split",
            showlegend=True,
        ))
        fig_bt.update_layout(
            title=f"Backtest: Actual vs Predicted — Store {STORE_ID} (last 20% held out)",
            xaxis_title="Date", yaxis_title="Units Sold",
        )
        st.plotly_chart(fig_bt, width="stretch")
        st.caption(
            f"Training cutoff: {bt['cutoff_date'].strftime('%d %b %Y')}. "
            "Dashed line = Prophet forecast. Shaded band = prediction interval."
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB: FORECAST & RISK
# ══════════════════════════════════════════════════════════════════════════════
with tab_forecast:
    st.header(f"What do we expect in the next {HORIZON} days?")
    st.markdown(
        f"""
        This forecast shows expected demand for **Store {STORE_ID}** 
        over the next **{HORIZON} days**.

        **Why this matters:** It allows teams to prepare inventory and staffing in advance.  
        The shaded band shows the prediction interval — the plausible range of outcomes.
        """
    )

    show_interval = st.toggle("Show prediction interval", value=True)

    with st.spinner(f"Generating {HORIZON}-day forecast..."):
        forecast_df = get_forecast(STORE_ID, HORIZON)

    forecast_series = forecast_df["yhat"]
    forecast_mean   = float(forecast_series.mean())
    forecast_std    = float(forecast_series.std())

    hist_series = series_full[-60:]

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=hist_series.index, y=hist_series.values,
        mode="lines", name="Historical Sales", line=dict(color="#1f77b4"),
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df["yhat"],
        mode="lines", name="Forecast", line=dict(color="#ff7f0e"),
    ))
    if show_interval:
        fig_forecast.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df["yhat_upper"],
            mode="lines", line=dict(width=0), showlegend=False,
        ))
        fig_forecast.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df["yhat_lower"],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(255, 127, 14, 0.2)",
            name="Prediction Interval",
        ))
    fig_forecast.update_layout(
        title=f"{HORIZON}-Day Sales Forecast — Store {STORE_ID}",
        xaxis_title="Date", yaxis_title="Units Sold",
    )
    st.plotly_chart(fig_forecast, width="stretch")

    demand_lo = int(forecast_mean - 1.5 * forecast_std)
    demand_hi = int(forecast_mean + 1.5 * forecast_std)
    st.warning(
        f"**Expected daily demand range:** {max(0, demand_lo):,} – {demand_hi:,} units  \n"
        f"This range reflects uncertainty and helps manage risk."
    )

    # ── Download forecast CSV ─────────────────────────────────────────────────
    st.markdown("**Export forecast data**")
    export_df = forecast_df.copy().reset_index()
    export_df.columns = ["Date", "Forecast", "Lower_Bound", "Upper_Bound"]
    export_df["Store"] = STORE_ID
    export_df["Date"]  = export_df["Date"].dt.strftime("%Y-%m-%d")

    st.download_button(
        label="Download forecast as CSV",
        data=export_df.to_csv(index=False),
        file_name=f"forecast_store_{STORE_ID}_{HORIZON}d.csv",
        mime="text/csv",
        help="Download the forecast table for use in Excel or your planning tools.",
    )
    st.caption("Columns: Date, Forecast (units), Lower Bound, Upper Bound, Store ID.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB: INVENTORY RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_inventory:
    st.header("How much inventory should we plan?")
    st.markdown(
        """
        Inventory is recommended as a **range** rather than a single number.

        **Why:** Demand is uncertain. A small safety buffer prevents stockouts,
        which are typically more costly than holding a little extra stock.
        """
    )

    # Ensure forecast is available (may have been cached from another tab)
    with st.spinner("Loading forecast..."):
        forecast_df   = get_forecast(STORE_ID, HORIZON)
        forecast_mean = float(forecast_df["yhat"].mean())
        forecast_std  = float(forecast_df["yhat"].std())

    weeks          = HORIZON / 7
    recommended_min= max(0, int(forecast_mean * weeks))
    recommended_max= int((forecast_mean + 1.2 * forecast_std) * weeks)

    i1, i2 = st.columns(2)
    i1.metric(
        "Minimum recommended",
        f"{recommended_min:,} units",
        help="Based on the forecast mean — covers expected demand with no buffer.",
    )
    i2.metric(
        "Maximum (with safety buffer)",
        f"{recommended_max:,} units",
        delta=f"+{recommended_max - recommended_min:,} buffer units",
        delta_color="off",
        help="Mean + 1.2× std dev — covers demand in most scenarios.",
    )

    st.success(
        f"**Recommended inventory for next {HORIZON} days:**  \n"
        f"Minimum: **{recommended_min:,} units**  \n"
        f"Maximum (with safety buffer): **{recommended_max:,} units**"
    )

    # Per-day breakdown table
    st.markdown("**Day-by-day forecast breakdown**")
    daily_df = forecast_df.copy().reset_index()
    daily_df.columns = ["Date", "Expected", "Lower", "Upper"]
    daily_df["Expected"] = daily_df["Expected"].clip(lower=0).round(0).astype(int)
    daily_df["Lower"]    = daily_df["Lower"].clip(lower=0).round(0).astype(int)
    daily_df["Upper"]    = daily_df["Upper"].clip(lower=0).round(0).astype(int)
    daily_df["Day"]      = daily_df["Date"].dt.strftime("%a %d %b")
    st.dataframe(
        daily_df[["Day", "Expected", "Lower", "Upper"]].set_index("Day"),
        width="stretch",
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB: SCENARIO SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_scenario:
    st.header("What if demand changes?")
    st.markdown(
        """
        Demand can shift due to promotions, holidays, or external factors.

        Use the slider below to see how inventory needs change under different demand scenarios.
        """
    )

    with st.spinner("Loading forecast..."):
        forecast_df   = get_forecast(STORE_ID, HORIZON)
        forecast_mean = float(forecast_df["yhat"].mean())
        forecast_std  = float(forecast_df["yhat"].std())

    demand_change_pct = st.slider(
        "Simulate demand change (%)", -30, 30, 0, 5,
        help="Positive = demand boost (promotion, holiday). Negative = demand drop (weather, competition).",
    )

    adjusted_mean = forecast_mean * (1 + demand_change_pct / 100)
    scenario_min  = max(0, int(adjusted_mean * (HORIZON / 7)))
    scenario_max  = int((adjusted_mean + 1.2 * forecast_std) * (HORIZON / 7))

    s1, s2, s3 = st.columns(3)
    s1.metric("Demand adjustment",  f"{demand_change_pct:+d}%")
    s2.metric("Adjusted min stock", f"{scenario_min:,} units",
              delta=f"{scenario_min - int(forecast_mean*(HORIZON/7)):+,} vs baseline")
    s3.metric("Adjusted max stock", f"{scenario_max:,} units")

    st.info(f"Under this scenario: stock **{scenario_min:,} – {scenario_max:,} units** "
            f"for the next {HORIZON} days.")

    # Visual comparison: baseline vs scenario
    scenario_df = pd.DataFrame({
        "Scenario":   ["Baseline", f"{demand_change_pct:+d}% demand shift"],
        "Min Stock":  [int(forecast_mean * (HORIZON / 7)), scenario_min],
        "Max Stock":  [int((forecast_mean + 1.2*forecast_std) * (HORIZON/7)), scenario_max],
    })
    fig_scen = px.bar(
        scenario_df.melt(id_vars="Scenario", var_name="Buffer", value_name="Units"),
        x="Scenario", y="Units", color="Buffer", barmode="group",
        title="Inventory Range: Baseline vs Scenario",
        text_auto=True,
    )
    st.plotly_chart(fig_scen, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB: ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_anomaly:
    st.header("Were there any unusual demand events?")
    st.markdown(
        """
        This checks whether recent sales deviated significantly from normal patterns.

        **Why this matters:** Unusual spikes or drops may indicate promotions,
        supply disruptions, or data quality issues that need investigation.

        **Method:** Z-score on STL residuals — flags any day more than 3 standard 
        deviations from the expected pattern.
        """
    )

    with st.spinner("Running anomaly detection..."):
        stl_result, _ = get_stl(STORE_ID)

    residuals = stl_result.resid.dropna()
    z_scores  = (residuals - residuals.mean()) / residuals.std()
    anomalies = z_scores[np.abs(z_scores) > 3]

    if anomalies.empty:
        st.success("No unusual demand events detected in this store's history.")
    else:
        st.error(f"Detected **{len(anomalies)}** unusual demand event(s).")

        # Enrich anomaly table with actual context
        anomaly_detail = pd.DataFrame({
            "Date":          anomalies.index,
            "Z-Score":       anomalies.values.round(2),
            "Actual Sales":  series_full.reindex(anomalies.index).values,
            "Expected (STL)":stl_result.trend.reindex(anomalies.index).round(0).values,
            "Direction":     np.where(anomalies.values > 0, "Spike (higher than expected)",
                                                             "Drop (lower than expected)"),
        }).sort_values("Date", ascending=False)
        anomaly_detail["Date"] = anomaly_detail["Date"].dt.strftime("%a %d %b %Y")

        st.dataframe(anomaly_detail.set_index("Date"), width="stretch")
        st.caption(
            "**Z-Score** = how many standard deviations the actual sales were from the STL trend. "
            "|Z| > 3 is flagged as anomalous. "
            "**Actual Sales** and **Expected** are in units."
        )

        # Plot anomalies on the full series
        fig_anom = go.Figure()
        fig_anom.add_trace(go.Scatter(
            x=series_display.index, y=series_display.values,
            mode="lines", name="Daily Sales", line=dict(color="#1f77b4"),
        ))
        # Only plot anomalies within the display range
        anom_in_range = anomalies[
            (anomalies.index >= display_start) & (anomalies.index <= display_end)
        ]
        if not anom_in_range.empty:
            fig_anom.add_trace(go.Scatter(
                x=anom_in_range.index,
                y=series_full.reindex(anom_in_range.index).values,
                mode="markers", name="Anomaly",
                marker=dict(color="red", size=10, symbol="x"),
            ))
        fig_anom.update_layout(
            title=f"Sales with Anomalies Highlighted — Store {STORE_ID}",
            xaxis_title="Date", yaxis_title="Units Sold",
        )
        st.plotly_chart(fig_anom, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB: FINAL INSIGHT
# ══════════════════════════════════════════════════════════════════════════════
with tab_insight:
    st.header("Final Business Insight")
    st.markdown(
        "This section synthesises the analysis into a **clear, actionable recommendation** "
        "for inventory and planning decisions."
    )

    with st.spinner("Generating insight..."):
        forecast_df    = get_forecast(STORE_ID, HORIZON)
        forecast_series= forecast_df["yhat"]
        insight        = generate_business_insight(series_full, forecast_series)

    st.success(insight)

    # ── Transparency: show how the insight was derived ─────────────────────────
    with st.expander("How was this insight generated?"):
        trend_slope  = float(np.polyfit(range(min(30, len(series_full))),
                                        series_full.values[-30:], 1)[0])
        trend_dir    = "upward" if trend_slope > 0 else "downward"
        fc_mean      = float(forecast_series.mean())
        hist_mean    = float(series_full.mean())
        vs_hist_pct  = (fc_mean - hist_mean) / hist_mean * 100 if hist_mean != 0 else 0

        stl_r, _     = get_stl(STORE_ID)
        residuals    = stl_r.resid.dropna()
        z_scores     = (residuals - residuals.mean()) / residuals.std()
        n_anomalies  = int((np.abs(z_scores) > 3).sum())

        st.markdown(
            f"""
            The insight above was derived from these signals:

            | Signal | Value | Interpretation |
            |--------|-------|----------------|
            | Recent trend (30d) | {trend_slope:+.1f} units/day | Demand is on an **{trend_dir}** trajectory |
            | Forecast mean | {fc_mean:,.0f} units/day | {vs_hist_pct:+.1f}% vs historical average |
            | Historical avg | {hist_mean:,.0f} units/day | Baseline reference |
            | Anomalies detected | {n_anomalies} | Unusual demand events in full history |
            | Forecast horizon | {HORIZON} days | Planning window |

            The `generate_business_insight()` function in `src/insight_engine.py` 
            uses these computed signals to construct a rule-based or model-based narrative.
            """
        )

st.markdown("---")
st.caption(
    "**Built by Sarthak Shandilya** · "
    "Tools: Python · Pandas · Statsmodels · Prophet · Plotly · Streamlit"
)

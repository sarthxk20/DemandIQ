import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from statsmodels.tsa.seasonal import STL

from src.data_loader import load_data
from src.prophet_model import prophet_forecast
from src.insight_engine import generate_business_insight


# -------------------------------------------------
# App config
# -------------------------------------------------
st.set_page_config(
    page_title="Retail Demand Forecasting & Risk Insights",
    layout="wide"
)

st.markdown(
    "<h1 style='font-size: 64px; margin-bottom: 0;'>DemandIQ</h1>",
    unsafe_allow_html=True
)
st.title("Retail Demand Forecasting & Risk Insights")

st.markdown(
    """
    This dashboard explains **how demand behaves**, **what to expect next**, and  
    **how to make better inventory decisions**.
    """
)

st.divider()

# -------------------------------------------------
# Load data
# -------------------------------------------------
train = load_data()

# -------------------------------------------------
# Sidebar navigation
# -------------------------------------------------
st.sidebar.header("Navigation")

st.sidebar.markdown(
    """
- [Executive Summary](#executive-summary)
- [Data Quality](#data-quality)
- [Sales Behavior](#sales-behavior)
- [What Changed](#what-changed)
- [Demand Drivers](#demand-drivers)
- [Model Comparison](#model-comparison)
- [Why Prophet](#why-prophet)
- [Forecast & Risk](#forecast-risk)
- [Inventory Recommendation](#inventory)
- [Scenario Simulation](#scenario)
- [Anomaly Detection](#anomaly)
- [Final Insight](#final-insight)
"""
)

# -------------------------------------------------
# Store selection
# -------------------------------------------------
st.sidebar.header("Store Selection")

store_ids = sorted(train["Store"].unique())
STORE_ID = st.sidebar.selectbox("Select Store ID", store_ids)

HORIZON = 14

store_df = train[train["Store"] == STORE_ID].copy()
store_df = store_df.sort_values("Date")
store_df.set_index("Date", inplace=True)

series = store_df["Sales"].asfreq("D", fill_value=0)

# -------------------------------------------------
# Executive Summary
# -------------------------------------------------
st.markdown("<div id='executive-summary'></div>", unsafe_allow_html=True)
st.header("Executive Summary")

st.markdown(
    """
    **In simple terms:**  
    Sales follow a **weekly rhythm** — some days consistently perform better than others.

    **Why this matters:**  
    Understanding this pattern allows teams to **plan ahead** instead of reacting late.

    **Impact:**  
    Seasonality-aware forecasting reduced errors by **~60%**, helping prevent
    stockouts and unnecessary overstocking.
    """
)

st.divider()

# -------------------------------------------------
# Data quality
# -------------------------------------------------
st.markdown("<div id='data-quality'></div>", unsafe_allow_html=True)
st.header("Data Quality Snapshot")

if store_df.empty:
    st.warning("No data available for this store.")
else:
    date_min = store_df.index.min()
    date_max = store_df.index.max()
    expected_dates = pd.date_range(start=date_min, end=date_max, freq="D")
    missing_count = len(expected_dates.difference(store_df.index))

    zero_sales_days = int((store_df["Sales"] == 0).sum())

    sales_values = store_df["Sales"]
    q1 = sales_values.quantile(0.25)
    q3 = sales_values.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        outlier_count = 0
    else:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((sales_values < lower) | (sales_values > upper)).sum())

    col_missing, col_zero, col_outlier = st.columns(3)
    col_missing.metric("Missing dates", missing_count)
    col_zero.metric("Zero-sales days", zero_sales_days)
    col_outlier.metric("Outlier days (IQR)", outlier_count)

    st.caption(
        "Missing dates are gaps in raw data before daily filling. "
        "Outliers use the IQR rule."
    )

st.divider()

# -------------------------------------------------
# Sales behavior
# -------------------------------------------------
st.markdown("<div id='sales-behavior'></div>", unsafe_allow_html=True)
st.header("How has this store been selling historically?")

st.markdown(
    """
    **What this chart shows:**  
    Daily sales over time.

    **What to notice:**  
    - Repeating ups and downs  
    - Predictable demand patterns  

    **Why this matters:**  
    Predictable demand enables reliable forecasting.
    """
)

fig_sales = px.line(
    series,
    title=f"Daily Sales Over Time — Store {STORE_ID}",
    labels={"value": "Units Sold", "index": "Date"}
)

st.plotly_chart(fig_sales, use_container_width=True)

st.divider()

# -------------------------------------------------
# What changed
# -------------------------------------------------
st.markdown("<div id='what-changed'></div>", unsafe_allow_html=True)
st.header("What changed recently?")

if len(series) < 14:
    st.info("Not enough history to summarize recent changes.")
else:
    last_7 = series[-7:].mean()
    prev_7 = series[-14:-7].mean()
    pct_7 = None if prev_7 == 0 else (last_7 - prev_7) / prev_7

    if len(series) >= 60:
        last_30 = series[-30:].mean()
        prev_30 = series[-60:-30].mean()
        pct_30 = None if prev_30 == 0 else (last_30 - prev_30) / prev_30
        change_30 = (
            "N/A"
            if pct_30 is None or np.isnan(pct_30)
            else f"{pct_30 * 100:+.1f}%"
        )
        change_30_line = (
            f"- Last 30 days: {last_30:,.0f} avg units/day vs prior 30 days "
            f"{prev_30:,.0f} ({change_30})."
        )
    else:
        change_30_line = (
            "- Last 30 days: not enough history (need 60 days of data)."
        )

    change_7 = (
        "N/A"
        if pct_7 is None or np.isnan(pct_7)
        else f"{pct_7 * 100:+.1f}%"
    )

    st.markdown(
        f"""
        **Summary:**  
        - Last 7 days: {last_7:,.0f} avg units/day vs prior 7 days {prev_7:,.0f} ({change_7}).  
        {change_30_line}
        """
    )

st.divider()

# -------------------------------------------------
# Demand drivers
# -------------------------------------------------
st.markdown("<div id='demand-drivers'></div>", unsafe_allow_html=True)
st.header("What is driving sales changes?")

st.markdown(
    """
    This section breaks sales into understandable components:

    - **Trend:** Long-term growth or decline  
    - **Weekly pattern:** Regular recurring behavior  
    - **Residual (Unexpected changes):** Unusual events such as promotions or disruptions  

    **Why this matters:**  
    Understanding these drivers helps select the right forecasting strategy.
    """
)

stl = STL(series, period=7, robust=True)
result = stl.fit()

decomp_df = pd.DataFrame({
    "Observed Sales": result.observed,
    "Trend (Long-Term Movement)": result.trend,
    "Weekly Pattern (Seasonality)": result.seasonal,
    "Residual (Unexpected Changes)": result.resid
})

fig_decomp = px.line(
    decomp_df,
    facet_row="variable",
    height=800,
    title="Breaking Down Sales Behavior"
)

# Clean up facet labels (remove vertical right-side text and place labels on the left)
fig_decomp.update_yaxes(title_text="")
fig_decomp.update_layout(
    legend_title_text="",
    margin=dict(l=160, r=40)
)
for annotation in fig_decomp.layout.annotations:
    if annotation.text.startswith("variable="):
        annotation.update(
            text=annotation.text.split("=", 1)[1],
            textangle=0,
            x=0,
            xanchor="right",
            yanchor="middle"
        )

st.plotly_chart(fig_decomp, use_container_width=True)

st.info(
    "**Key takeaway:** Weekly seasonality explains most of the sales variation. "
    "Residuals highlight unusual or one-off events."
)

st.divider()

# -------------------------------------------------
# Model comparison
# -------------------------------------------------
st.markdown("<div id='model-comparison'></div>", unsafe_allow_html=True)
st.header("Which forecasting approach works best?")

st.markdown(
    """
    **How to read this chart:**  
    - Each bar represents a forecasting approach  
    - **Lower bars indicate better accuracy**

    **Why this matters:**  
    More accurate forecasts reduce inventory risk.
    """
)

comparison_df = pd.DataFrame({
    "Model": ["Naive", "Moving Average", "ARIMA", "SARIMA", "Prophet"],
    "Average Error (MAE)": [1993.86, 1519.18, 1412.63, 784.21, 778.15]
})

fig_comp = px.bar(
    comparison_df,
    x="Model",
    y="Average Error (MAE)",
    text_auto=True,
    title="Forecast Accuracy Comparison (Lower is Better)"
)

st.plotly_chart(fig_comp, use_container_width=True)

st.success(
    "Seasonality-aware models reduce forecasting errors by approximately **60%**."
)

st.divider()

# -------------------------------------------------
# Why Prophet
# -------------------------------------------------
st.markdown("<div id='why-prophet'></div>", unsafe_allow_html=True)
st.header("Why was Prophet chosen?")

st.markdown(
    """
    Prophet performs on par with advanced statistical models while being:

    - Easier to maintain  
    - Faster to retrain  
    - More robust to changing demand patterns  

    This makes it well suited for **real-world business use**.
    """
)

st.divider()

# -------------------------------------------------
# Forecast & risk
# -------------------------------------------------
st.markdown("<div id='forecast-risk'></div>", unsafe_allow_html=True)
st.header("What do we expect in the next 14 days?")

st.markdown(
    """
    This forecast shows expected demand over the next two weeks.

    **Why this matters:**  
    It allows teams to prepare inventory and staffing **in advance**.

    The shaded band shows the prediction interval to reflect uncertainty.
    """
)

show_interval = st.toggle("Show prediction interval", value=True)

forecast_df = prophet_forecast(
    series,
    horizon=HORIZON,
    return_intervals=True
)

forecast_series = forecast_df["yhat"]
forecast_mean = forecast_series.mean()
forecast_std = forecast_series.std()

hist_series = series[-60:]

fig_forecast = go.Figure()
fig_forecast.add_trace(
    go.Scatter(
        x=hist_series.index,
        y=hist_series.values,
        mode="lines",
        name="Historical Sales"
    )
)
fig_forecast.add_trace(
    go.Scatter(
        x=forecast_df.index,
        y=forecast_df["yhat"],
        mode="lines",
        name="Forecast"
    )
)
if show_interval:
    fig_forecast.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False
        )
    )
    fig_forecast.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df["yhat_lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.2)",
            name="Prediction Interval"
        )
    )
fig_forecast.update_layout(
    title="14-Day Sales Forecast with Prediction Interval",
    xaxis_title="Date",
    yaxis_title="Units Sold"
)

st.plotly_chart(fig_forecast, use_container_width=True)

st.warning(
    f"""
    **Expected daily demand range:**  
    {int(forecast_mean - 1.5*forecast_std):,} – {int(forecast_mean + 1.5*forecast_std):,} units  

    This range reflects uncertainty and helps manage risk.
    """
)

st.divider()

# -------------------------------------------------
# Inventory recommendation
# -------------------------------------------------
st.markdown("<div id='inventory'></div>", unsafe_allow_html=True)
st.header("How much inventory should we plan?")

st.markdown(
    """
    Inventory is recommended as a **range** rather than a single number.

    **Why:**  
    Under-stocking is typically more costly than holding a small buffer.
    """
)

recommended_min = int(forecast_mean * (HORIZON / 7))
recommended_max = int((forecast_mean + 1.2 * forecast_std) * (HORIZON / 7))

st.success(
    f"""
    **Recommended inventory for next {HORIZON} days:**
    - Minimum: {recommended_min:,} units
    - Maximum (with safety buffer): {recommended_max:,} units
    """
)

st.divider()

# -------------------------------------------------
# Scenario simulation
# -------------------------------------------------
st.markdown("<div id='scenario'></div>", unsafe_allow_html=True)
st.header("What if demand changes?")

st.markdown(
    """
    Demand can shift due to promotions, holidays, or external factors.

    Use the slider to see how inventory needs change under different scenarios.
    """
)

demand_change_pct = st.slider(
    "Simulate demand change (%)",
    -30, 30, 0, 5
)

adjusted_mean = forecast_mean * (1 + demand_change_pct / 100)

scenario_min = int(adjusted_mean * (HORIZON / 7))
scenario_max = int((adjusted_mean + 1.2 * forecast_std) * (HORIZON / 7))

st.info(
    f"Adjusted inventory range: {scenario_min:,} – {scenario_max:,} units"
)

st.divider()

# -------------------------------------------------
# Anomaly detection
# -------------------------------------------------
st.markdown("<div id='anomaly'></div>", unsafe_allow_html=True)
st.header("Were there any unusual demand events?")

st.markdown(
    """
    This checks whether recent sales deviated significantly from normal patterns.

    **Why this matters:**  
    Unusual spikes or drops may indicate promotions, disruptions, or data issues.
    """
)

residuals = result.resid.dropna()
z_scores = (residuals - residuals.mean()) / residuals.std()
anomalies = z_scores[np.abs(z_scores) > 3]

if anomalies.empty:
    st.success("No unusual demand events detected recently.")
else:
    st.error(f"Detected {len(anomalies)} unusual demand events.")
    st.dataframe(anomalies.rename("residual").tail(10))

st.divider()

# -------------------------------------------------
# Final insight
# -------------------------------------------------
st.markdown("<div id='final-insight'></div>", unsafe_allow_html=True)
st.header("Final Business Insight")

st.markdown(
    """
    **In summary:**  
    This system transforms historical data and forecasts into
    **clear, actionable guidance** for inventory and planning decisions.
    """
)

insight = generate_business_insight(series, forecast_series)
st.success(insight)

st.markdown("---")

st.caption(
    """
    **Built by Sarthak Shandilya**  
    Tools used: Python, Pandas, Statsmodels, Prophet, Plotly, Streamlit
    """
)



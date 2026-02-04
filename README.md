# DemandIQ
## Retail Demand Forecasting & Risk Insights

DemandIQ is an end-to-end retail demand forecasting and risk insights system that transforms historical sales data into clear, actionable inventory and planning decisions.

The project is designed as a business-friendly decision support dashboard, helping non-technical stakeholders understand demand behavior, anticipate short-term sales, manage uncertainty, and plan inventory proactively.

---

## Why DemandIQ?

Retail demand is rarely random — it follows predictable patterns, especially weekly cycles.
When these patterns are ignored, businesses are forced into reactive decisions, leading to stockouts or overstocking.

DemandIQ addresses this by:
- Explaining how demand behaves
- Forecasting what is likely to happen next
- Quantifying risk and uncertainty
- Translating predictions into practical inventory guidance

---

## Key Features

- Demand Behavior Analysis  
  Understand historical sales trends and weekly patterns.

- Demand Driver Decomposition  
  Break sales into trend, seasonality, and residual (unexpected changes).

- Model Comparison & Selection  
  Evaluate multiple forecasting approaches and justify model choice.

- Short-Term Demand Forecasting  
  Generate 14-day forecasts using seasonality-aware models.

- Risk & Uncertainty Estimation  
  Communicate expected demand ranges instead of single-point predictions.

- Inventory Recommendations  
  Convert forecasts into actionable inventory planning ranges.

- Scenario Simulation (What-If Analysis)  
  Test how inventory needs change under demand increases or decreases.

- Anomaly Detection  
  Identify unusual demand events using residual analysis.

- Automated Business Insights  
  Summarize findings in plain language for decision-makers.

---

## How It Works

1. Historical sales data is analyzed to identify trends and weekly patterns
2. Sales are decomposed into:
   - Trend (long-term movement)
   - Seasonality (weekly cycles)
   - Residuals (unexpected changes)
3. Multiple forecasting models are evaluated
4. A seasonality-aware model is selected
5. Forecasts are combined with uncertainty estimates
6. Results are translated into inventory and planning recommendations

---

## Application Overview

The Streamlit dashboard guides users through a clear narrative:

1. Executive Summary
2. Historical Sales Behavior
3. Demand Drivers
4. Model Performance Comparison
5. Forecast & Risk Outlook
6. Inventory Recommendations
7. Scenario Simulation
8. Anomaly Detection
9. Final Business Insight

All sections include plain-English explanations so the dashboard can be used by non-technical audiences.

---

## Tech Stack

- Python
- Pandas, NumPy
- Statsmodels
- Prophet
- Scikit-learn
- Plotly
- Streamlit

---

## Project Structure


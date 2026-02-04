# DemandIQ
## Retail Demand Forecasting & Risk Insights

DemandIQ is an end-to-end retail demand forecasting and risk insights system that transforms historical sales data into clear, actionable inventory and planning decisions.

The project is built as a business-friendly decision support dashboard, allowing non-technical stakeholders to understand demand behavior, anticipate short-term sales, manage uncertainty, and plan inventory proactively.

---

## Why DemandIQ?

Retail demand is rarely random. It follows predictable patterns, especially weekly cycles.  
When these patterns are ignored, businesses are forced into reactive decisions, leading to stockouts or overstocking.

DemandIQ addresses this problem by:
- Explaining how demand behaves
- Forecasting what is likely to happen next
- Quantifying risk and uncertainty
- Translating predictions into practical inventory guidance

---

## Key Features

- Demand Behavior Analysis  
  Analyze historical sales trends and recurring weekly patterns.

- Demand Driver Decomposition  
  Break sales into trend, seasonality, and residual (unexpected changes).

- Model Comparison & Selection  
  Compare multiple forecasting approaches and justify the final model choice.

- Short-Term Demand Forecasting  
  Generate 14-day forecasts using seasonality-aware models.

- Risk & Uncertainty Estimation  
  Communicate expected demand ranges instead of single-point predictions.

- Inventory Recommendations  
  Convert forecasts into actionable inventory planning ranges.

- Scenario Simulation (What-If Analysis)  
  Simulate demand increases or decreases and observe inventory impact.

- Anomaly Detection  
  Identify unusual demand events using residual analysis.

- Automated Business Insights  
  Generate plain-language summaries for decision-makers.

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

The Streamlit dashboard follows a clear, narrative-driven flow:

1. Executive Summary  
2. Historical Sales Behavior  
3. Demand Drivers  
4. Model Performance Comparison  
5. Forecast & Risk Outlook  
6. Inventory Recommendations  
7. Scenario Simulation  
8. Anomaly Detection  
9. Final Business Insight  

All sections include plain-English explanations to ensure usability for non-technical audiences.

---

## Tech Stack

- Python
- Pandas
- NumPy
- Statsmodels
- Prophet
- Scikit-learn
- Plotly
- Streamlit

---

## Data Availability

Raw datasets are not included in this repository due to size and licensing considerations.

The project is designed to work with publicly available retail sales data.  
Instructions for obtaining and placing the data locally are documented in `data_loader.py`.

---

## How to Run Locally

1. Clone the repository

git clone https://github.com/sarthxk20/DemandIQ.git  
cd DemandIQ  

2. Install dependencies

pip install -r requirements.txt  

3. Run the application

streamlit run app.py  

---

## Author

Sarthak Shandilya  
Data Science & Machine Learning  

GitHub: https://github.com/sarthxk20  

---

## Final Note

DemandIQ demonstrates how data science goes beyond building models and into decision-making, risk awareness, and real-world business impact.

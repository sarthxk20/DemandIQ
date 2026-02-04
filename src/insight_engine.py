import numpy as np
import pandas as pd


def generate_business_insight(
    series: pd.Series,
    forecast: pd.Series
) -> str:
    """
    Generate business-friendly insights without external LLMs.
    """

    avg_sales = series.mean()
    recent_avg = series[-14:].mean()
    forecast_avg = forecast.mean()

    volatility = series[-14:].std()
    change_pct = ((forecast_avg - recent_avg) / recent_avg) * 100

    insights = []

    # Trend insight
    if change_pct > 10:
        insights.append(
            f"Demand is expected to increase by approximately {change_pct:.1f}% over the next two weeks."
        )
    elif change_pct < -10:
        insights.append(
            f"Demand is expected to decline by approximately {abs(change_pct):.1f}% over the next two weeks."
        )
    else:
        insights.append(
            "Demand levels are expected to remain relatively stable over the next two weeks."
        )

    # Volatility insight
    if volatility > 0.5 * avg_sales:
        insights.append(
            "Sales show high variability, suggesting potential promotions or irregular demand events."
        )
    else:
        insights.append(
            "Sales patterns appear stable with predictable weekly behavior."
        )

    # Recommendation
    insights.append(
        "Recommended action: plan inventory using weekly demand patterns and monitor sudden deviations closely."
    )

    return " ".join(insights)
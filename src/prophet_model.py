import pandas as pd
from prophet import Prophet


def prophet_forecast(
    series: pd.Series,
    horizon: int,
    return_intervals: bool = False
):
    """
    Fit Prophet model and forecast future values.
    """
    # Prophet requires specific column names
    df = series.reset_index()
    df.columns = ["ds", "y"]

    model = Prophet(
        weekly_seasonality=True,
        daily_seasonality=False,
        yearly_seasonality=False
    )

    model.fit(df)

    future = model.make_future_dataframe(periods=horizon, freq="D")
    forecast = model.predict(future)

    # Return only the forecast horizon
    forecast_horizon = forecast.loc[
        forecast.index[-horizon:],
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].set_index("ds")

    if return_intervals:
        return forecast_horizon

    return pd.Series(
        forecast_horizon["yhat"].values,
        index=forecast_horizon.index
    )

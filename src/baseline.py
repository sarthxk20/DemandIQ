import pandas as pd
import numpy as np


def naive_forecast(series: pd.Series, horizon: int):
    """
    Forecast using the last observed value.
    """
    last_value = series.iloc[-1]
    forecast = np.repeat(last_value, horizon)

    return pd.Series(forecast)


def moving_average_forecast(series: pd.Series, horizon: int, window: int = 7):
    """
    Forecast using simple moving average.
    """
    avg = series.iloc[-window:].mean()
    forecast = np.repeat(avg, horizon)

    return pd.Series(forecast)
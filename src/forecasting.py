import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def arima_forecast(
    series: pd.Series,
    horizon: int,
    order: tuple = (1, 1, 1)
):
    """
    Fit ARIMA model and forecast future values.
    """
    model = ARIMA(series, order=order)
    fitted_model = model.fit()

    forecast = fitted_model.forecast(steps=horizon)

    return forecast


def sarima_forecast(
    series: pd.Series,
    horizon: int,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 7)
):
    """
    Fit SARIMA model and forecast future values.
    """
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted_model = model.fit(disp=False)
    forecast = fitted_model.forecast(steps=horizon)

    return forecast
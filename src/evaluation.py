import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error



def walk_forward_validation(
    series: pd.Series,
    forecast_func,
    horizon: int,
    initial_train_size: int,
    step: int = 1,
    **forecast_kwargs
):
    """
    Perform walk-forward validation on a time series.
    """
    errors = []

    for start in range(
        initial_train_size,
        len(series) - horizon,
        step
    ):
        train = series.iloc[:start]
        test = series.iloc[start:start + horizon]

        forecast = forecast_func(
            train,
            horizon=horizon,
            **forecast_kwargs
        )

        mae = mean_absolute_error(test, forecast)
        mse = mean_squared_error(test, forecast)
        rmse = np.sqrt(mse)


        errors.append({
            "train_end": train.index[-1],
            "MAE": mae,
            "RMSE": rmse
        })

    return pd.DataFrame(errors)
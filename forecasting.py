# Reusable Forecasting Functions

import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


# ARIMA FUNCTION

def train_arima(train_series, order=(2,1,2)):
    model = ARIMA(train_series, order=order)
    result = model.fit()
    return result


def evaluate_arima(model, test_series):
    forecast = model.forecast(steps=len(test_series))
    mape = mean_absolute_percentage_error(test_series, forecast)
    rmse = np.sqrt(mean_squared_error(test_series, forecast))
    accuracy = (1 - mape) * 100
    return forecast, accuracy, rmse


# PROPHET FUNCTION

def train_prophet(train_df, cps=0.1):
    model = Prophet(
        yearly_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=cps
    )
    model.fit(train_df)
    return model


def evaluate_prophet(model, test_df):
    future = model.make_future_dataframe(periods=len(test_df), freq='MS')
    forecast = model.predict(future)

    forecast_test = forecast[['ds','yhat']].tail(len(test_df))
    merged = test_df.merge(forecast_test, on='ds')

    mape = mean_absolute_percentage_error(merged['y'], merged['yhat'])
    rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
    accuracy = (1 - mape) * 100

    return forecast, accuracy, rmse


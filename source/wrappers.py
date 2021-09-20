import warnings

import numpy as np
import pandas as pd
import pmdarima as pm
from fbprophet import Prophet
from pmdarima.arima import auto_arima
from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings('ignore')

from ts_cv import smae_score


class WrapperHoltWinters:
    def __init__(self):
        pass

    def fit(self, data):
        """
        Fit function for Exponential Smoothing Holt-Winters model with additive trend and seasonality. Seasonality
        period equals 6 (a week with one day off without observations).

        :param data: time series in data frame format with necessary column ['y'], where 'y' represents values
        :return: None
        """
        self.model = ExponentialSmoothing(data['y'], trend='add', damped_trend=True,
                                          seasonal='add', seasonal_periods=6).fit()
        pass

    def predict(self, data, horizon):
        """
        Predict function for Exponential Smoothing Holt-Winters model with additive trend and seasonality. Seasonality
        period equals 6 (a week with one day off without observations).

        :param data: time series in data frame format with necessary column ['y'], where 'y' represents values
        :param horizon: forecast horizon (periods in the future to predict)
        :return: data frame format column ['yhat'] with predicted values for the chosen time horizon
        """
        data = data.reset_index(drop=True)
        pred = self.model.predict(start=data['y'].index[0], end=data['y'].index[-1] + horizon)
        pred = pd.DataFrame(pred)
        pred = pred[-(horizon):]
        pred.columns = ['yhat']
        forecast = pred.reset_index(drop=True)
        return forecast


class WrapperAutoArima:
    def __init__(self):
        pass

    def fit(self, data):
        """
        Fit function for auto ARIMA model with auto tuning of parameters:
            * the order (or number of time lags) of the auto-regressive (“AR”) model **p** varies between 1 and 4
            * the order of the moving-average (“MA”) model **q** varies between 1 and 4
            * the order of the auto-regressive portion of the seasonal model **P** varies between 1 and 4
            * the order of the moving-average portion of the seasonal mode **Q** varies between 1 and 4
            * the maximum number of non-seasonal differences **d** varies between 1 and 4
            * the order of the seasonal differencing **D** varies between 1 and 5

        :param data: time series in data frame format with necessary column ['y'], where 'y' represents values
        :return: None
        """
        data_y = data['y']
        self.model = pm.auto_arima(data_y, m=1,
                                   start_p=1, start_q=1, start_P=1, start_Q=1,
                                   max_p=4, max_q=4, max_P=4, max_Q=4,
                                   d=0, max_d=4, D=1, max_D=5,
                                   seasonal=True,
                                   stepwise=False,
                                   suppress_warnings=True,
                                   n_jobs=-1,
                                   error_action='ignore',
                                   trace=False)
        pass

    def predict(self, data=None, horizon):
        """
        Predict function for auto ARIMA model.

        :param data: time series in data frame format with necessary column ['y'], where 'y' represents values
        :param horizon: forecast horizon (periods in the future to predict)
        :return: data frame format column ['yhat'] with predicted values for the chosen time horizon
        """
        pred = self.model.predict(n_periods=horizon, return_conf_int=False)
        pred = pd.DataFrame(pred)
        pred.columns = ['yhat']
        forecast = pred.reset_index(drop=True)
        return forecast


class WrapperProphet:
    def __init__(self):
        pass

    def fit(self, data, cv=3, horizon=31):
        """
        Fit function for Prophet model from Facebook with tuning of:
            * seasonality_mode - additive and multiplicative
            * weekly_seasonality - witout weekly seasonality and with one using a Fourier order 2, 3, 5
            * monthly_seasonality - without and with using a Fourier order 2, 3
            * yearly_seasonality and daily_seasonality are disabled
        The best set of parameters is chosen according to the sMAE metric value (MAE to mean).

        :param data: time series in data frame format with necessary columns ['ds', 'y'], where 'ds' represents datetime
        and 'y' is column of observations
        :param cv: number of cross-validation folds (historically simulated forecasts)
        :param horizon: number of periods to predict, days (taking into consideration calendar days)
        :return: None
        """
        if cv == 0:
            self.model = Prophet()
            self.model.fit(data)
        else:
            params = ParameterGrid({'seasonality_mode': ('multiplicative', 'additive'),
                                    'monthly_seasonality': [0, 2, 3],
                                    'weekly_seasonality': [0, 2, 3]})
            end_dt = data['ds'].max()
            result = []
            for param in params:
                smae_iter = []
                for cv_iter in range(cv):
                    model = Prophet(weekly_seasonality=param['weekly_seasonality'],
                                    daily_seasonality=False,
                                    yearly_seasonality=False,
                                    seasonality_mode=param['seasonality_mode'],
                                    )
                    if param['monthly_seasonality'] != 0:
                        model.add_seasonality(name='monthly', period=30.5, fourier_order=param['monthly_seasonality'])
                    test_end_dt = end_dt - pd.Timedelta(days=horizon) * cv_iter
                    test_start_dt = test_end_dt - pd.Timedelta(days=horizon)
                    model.fit(data[data['ds'] < test_start_dt])
                    future = pd.DataFrame(data[(data['ds'] >= test_start_dt) & (data['ds'] < test_end_dt)]).reset_index(
                        drop=True)
                    forecast = model.predict(future)

                    dates = forecast[(forecast['ds'] >= test_start_dt) & \
                                     (forecast['ds'] < test_end_dt)]['ds']

                    pred = forecast[forecast['ds'].isin(dates)]['yhat'].reset_index(drop=True)
                    fact = data[data['ds'].isin(dates)]['y']
                    smae_iter.append(smae_score(fact, pred))
                    res = {'seasonality_mode': param['seasonality_mode'],
                           'weekly_seasonality': param['weekly_seasonality'],
                           'monthly_seasonality': param['monthly_seasonality'],
                           'sMAE': np.mean(smae_iter)}
                    result.append(res)
            best_result = pd.DataFrame(result).sort_values('sMAE').head(1)
            self.model = Prophet(weekly_seasonality=param['weekly_seasonality'],
                                 daily_seasonality=False,
                                 yearly_seasonality=False,
                                 seasonality_mode=best_result['seasonality_mode'].values[0])
            if best_result['monthly_seasonality'].values[0] != 0:
                self.model.add_seasonality(name='monthly', period=30.5,
                                           fourier_order=best_result['monthly_seasonality'].values[0])
            self.model.fit(data)
        pass

    def predict(self, data, horizon):
        """
        Predict function for Prophet model.

        :param data: time series in data frame format with necessary column ['y'], where 'y' represents values
        :param horizon: forecast horizon (periods in the future to predict)
        :return: data frame with columns ['ds', 'yhat'] with predicted values for the chosen time horizon
        """
        future = self.model.make_future_dataframe(periods=horizon)
        forecast = self.model.predict(future)
        forecast = forecast[-horizon:][['ds', 'yhat']].reset_index(drop=True)
        return forecast


class WrapperBasicProphet:
    def __init__(self):
        self.model = Prophet()
        pass

    def fit(self, data):
        self.model = Prophet()
        self.model.fit(data)
        pass

    def predict(self, data, horizon):
        future = self.model.make_future_dataframe(periods=horizon)
        forecast = self.model.predict(future)
        forecast = forecast[-horizon:][['ds', 'yhat']].reset_index(drop=True)
        return forecast

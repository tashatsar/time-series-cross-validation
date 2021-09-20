import numpy as np
import pandas as pd

import statistics as stat

from sklearn.metrics import mean_absolute_error


def smae_score(fact, predict):
    """
    Scaled mean absolute error (MAE/mean) calculation between two columns of data: observed and forecasted values

    :param fact: observed values
    :param predict: predicted future values
    :return: sMAE
    """
    fact = np.array(fact)
    predict = np.array(predict)
    return mean_absolute_error(fact, predict) / fact.mean()


def mape_score(fact, predict):
    """
    Mean absolute percentage error MAPE calculation between two columns of data: observed and forecasted values

    :param fact: observed values
    :param predict: predicted future values
    :return: MAPE
    """
    fact = np.array(fact)
    predict = np.array(predict)
    if len(predict) == 0 or len(fact) == 0:
        return -1
    mape = np.mean(np.abs((fact - predict) / fact))
    return mape


def ts_cv_metrics(data, model, cv_folds=3, horizon=31):
    """
    Forecasting function with support of predictive models with fit and predict methods.
    NB-The number of folds of cross-validation and the forecast horizon are limited by the number of observations (and common sense)

    :param data: time series in data frame format with necessary columns ['ds', 'y'], where 'ds' represents datetime
    and 'y' is column of observations
    :param model: predictive models, fit and predict methods are required
    :param cv_folds: number of cross-validation folds (historically simulated forecasts)
    :param horizon: number of periods to predict, days (taking into consideration calendar days)
    :return: mean sMAE and MAPE within all CVs
    """
    result_smae = []
    result_mape = []
    end_dt = data['ds'].max()
    for cv_iter in range(cv_folds):
        test_end_dt = end_dt - pd.Timedelta(days=horizon) * cv_iter
        test_start_dt = test_end_dt - pd.Timedelta(days=horizon)
        train_df = data[(data['ds'] < test_start_dt)]
        test_df = data[(data['ds'] >= test_start_dt) & (data['ds'] < test_end_dt)]

        model.fit(train_df)
        if 'prophet' in (str(model)).lower():
            pred = model.predict(train_df, horizon=horizon)
            pred = pred[pred['ds'].isin(test_df['ds'])]
        else:
            pred = model.predict(train_df, horizon=len(test_df))
        smae = smae_score(test_df['y'], pred['yhat'])
        mape = mape_score(test_df['y'], pred['yhat'])
        result_smae.append(smae)
        result_mape.append(mape)

    mean_smae = stat.mean(result_smae)

    mean_mape = stat.mean(result_mape)

    return mean_smae, mean_mape, model.__class__.__name__[7:]

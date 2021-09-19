# time-series-forecasting-using-cross-validation
TS forecasting using CV. Bonuses: Facebook Prophet parameters tuning, TS missing values imputation.

Time series forecasting using modified dataset of [Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only/data):
* 20 stores without decomposition by item
* missing values of holidays (differs by store), weekends (differs by time period)
* 25% of time series have a significantly long missing periog of time

Approach for forecasting:
1. Descriptive analysis, discovery of days-off, holidays, missing values and missing time periods. 
2. Missing values imputation using weekly seasonality.
3. Division of time series into two groups: complete time series with big amount of observations and short time series with missing period. 
4. Testing on time series cross-validation three different models: Facebook Prophet (with parameters tuning on TS CV), Holt-Winters and auto ARIMA and for two groups of time series.
5. Forecasting for 07 month of 2017 using the best model selected at the previous step.  

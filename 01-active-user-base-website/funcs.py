import numpy as np
import pandas as pd
from pandas import Series

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import statsmodels
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

def test_stationarity(series, window=12):

    """
    Test stationarity of a time series.

    Arg:

    series: pandas.Series. The time series to be tested.

    window: the window time to calculate the rolling mean and rolling standard deviation.

    Returns:

    str: wether the series is stationary or not.

    """

    if not isinstance(series, pd.Series):
        series = pd.Series(data=series)    

    else:
        pass
    
    rol_mean = series.rolling(window=window).mean()
    rol_std = series.rolling(window=window).std()

    plt.plot(series, color="blue", label="Raw")
    plt.plot(rol_mean, color="red", label="Rolling Mean")
    plt.plot(rol_std, color='black', label="Rolling Std")
    plt.legend()
    plt.show()

    plot_acf(x=series, ax=plt.gca(), lags=13)
    plt.show()


    adf_test = np.array(adfuller(x=series, autolag='AIC'))

    df_adf_test = pd.Series(data=np.r_[adf_test[[0,1,2,3]], [adf_test[4].get('5%')]],
                           index=['statistic', 'p-value', 'lags', 'used_observations', 'critical_value_5%'])

    print(df_adf_test)
    
    p_value = adf_test[1]

    print('\nDickey-Fuller Test Results:')
    if p_value > 0.05:
        print('\nConclusion:\nNo evidence to reject H0. The series is NOT stationary.')
    
    else:
        print('\nConclusion:\nReject H0. The series IS stationary.')


def diff(series, period = 1):

    series = series.to_numpy()

    s = series[:-period] - series[period:]

    return pd.Series(s)
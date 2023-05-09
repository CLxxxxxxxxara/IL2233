import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# AR(1)
ar1 = np.array([1,-0.8])
ma1 = np.array([1])
AR1MA0 = ArmaProcess(ar1, ma1)
AR1MA0_series = AR1MA0.generate_sample(nsample=1000)

# MA(1)
ar1 = np.array([1])
ma1 = np.array([1,0.7])
AR0MA1 = ArmaProcess(ar1, ma1)
AR0MA1_series = AR0MA1.generate_sample(nsample=1000)

# ARMA(1,1) (1)
ar1 = np.array([1,-0.8])
ma1 = np.array([1,0.7])
AR1MA1_1 = ArmaProcess(ar1, ma1)
AR1MA1_1_series = AR1MA1_1.generate_sample(nsample=1000)

# ARMA(1,1) (2)
ar1 = np.array([1,0.8])
ma1 = np.array([1,-0.7])
AR1MA1_2 = ArmaProcess(ar1, ma1)
AR1MA1_2_series = AR1MA1_2.generate_sample(nsample=1000)

# Plotting time series
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0, 0].plot(AR1MA0_series)
axs[0, 0].set_title('AR(1)')
axs[0, 1].plot(AR0MA1_series)
axs[0, 1].set_title('MA(1)')
axs[1, 0].plot(AR1MA1_1_series)
axs[1, 0].set_title('ARMA(1,1) (1)')
axs[1, 1].plot(AR1MA1_2_series)
axs[1, 1].set_title('ARMA(1,1) (2)')
plt.show()

# Function to perform ADF test and print results
def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'{key}: {value}')
    if result[1] > 0.05:
        print('The series is non-stationary.')
    else:
        print('The series is stationary.')

# Applying ADF test on each time series
print('AR(1):')
adf_test(AR1MA0_series)
print('MR(1):')
adf_test(AR0MA1_series)
print('ARMR(1,1)(1):')
adf_test(AR1MA1_1_series)
print('ARMR(1,1)(2):')
adf_test(AR1MA1_2_series)


# Check if AR stationary and invertible
print(AR1MA0.isstationary)
print(AR1MA0.isinvertible)


# Check if MA stationary and invertible
print(AR0MA1.isstationary)
print(AR0MA1.isinvertible)


# Check if ARMA1 stationary and invertible
print(AR1MA1_1.isstationary)
print(AR1MA1_1.isinvertible)


# Check if ARMA2 stationary and invertible
print(AR1MA1_2.isstationary)
print(AR1MA1_2.isinvertible)



import seaborn as sns

# Plot histogram, density plot, and box plot for each time series
fig, axs = plt.subplots(4, 3, figsize=(10, 12))
sns.histplot(AR1MA0_series, ax=axs[0, 0])
sns.kdeplot(AR1MA0_series, ax=axs[0, 1])
sns.boxplot(AR1MA0_series, ax=axs[0, 2])
sns.histplot(AR0MA1_series, ax=axs[1, 0])
sns.kdeplot(AR0MA1_series, ax=axs[1, 1])
sns.boxplot(AR0MA1_series, ax=axs[1, 2])
sns.histplot(AR1MA1_1_series, ax=axs[2, 0])
sns.kdeplot(AR1MA1_1_series, ax=axs[2, 1])
sns.boxplot(AR1MA1_1_series, ax=axs[2, 2])
sns.histplot(AR1MA1_2_series, ax=axs[3, 0])
sns.kdeplot(AR1MA1_2_series, ax=axs[3, 1])
sns.boxplot(AR1MA1_2_series, ax=axs[3, 2])
axs[0, 0].set_title('AR(1) histogram')
axs[0, 1].set_title('AR(1) density plot')
axs[0, 2].set_title('AR(1) box plot')
axs[1, 0].set_title('MA(1) histogram')
axs[1, 1].set_title('MA(1) density plot')
axs[1, 2].set_title('MA(1) box plot')
axs[2, 0].set_title('ARMA(1,1)(1) histogram ')
axs[2, 1].set_title('ARMA(1,1)(1) density plot ')
axs[2, 2].set_title('ARMA(1,1)(1) box plot ')
axs[3, 0].set_title('ARMA(1,1)(2) histogram ')
axs[3, 1].set_title('ARMA(1,1)(2) density plot ')
axs[3, 2].set_title('ARMA(1,1)(2) box plot ')
plt.tight_layout()
plt.show()


# Plot lag-1 and lag-2 for each time series
fig, axs = plt.subplots(4, 2, figsize=(10, 12))
pd.plotting.lag_plot(pd.Series(AR1MA0_series), lag=1, ax=axs[0, 0])
pd.plotting.lag_plot(pd.Series(AR1MA0_series), lag=2, ax=axs[0, 1])
pd.plotting.lag_plot(pd.Series(AR0MA1_series), lag=1, ax=axs[1, 0])
pd.plotting.lag_plot(pd.Series(AR0MA1_series), lag=2, ax=axs[1, 1])
pd.plotting.lag_plot(pd.Series(AR1MA1_1_series), lag=1, ax=axs[2, 0])
pd.plotting.lag_plot(pd.Series(AR1MA1_1_series), lag=2, ax=axs[2, 1])
pd.plotting.lag_plot(pd.Series(AR1MA1_2_series), lag=1, ax=axs[3, 0])
pd.plotting.lag_plot(pd.Series(AR1MA1_2_series), lag=2, ax=axs[3, 1])
axs[0, 0].set_title('AR(1) lag-1 plot')
axs[0, 1].set_title('AR(1) lag-2 plot')
axs[1, 0].set_title('MA(1) lag-1 plot')
axs[1, 1].set_title('MA(1) lag-2 plot')
axs[2, 0].set_title('ARMA(1,1) lag-1 plot (1)')
axs[2, 1].set_title('ARMA(1,1) lag-2 plot (1)')
axs[3, 0].set_title('ARMA(1,1) lag-1 plot (2)')
axs[3, 1].set_title('ARMA(1,1) lag-2 plot (2)')
plt.tight_layout()
plt.show()

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Plot ACF and PACF for each time series
fig, axs = plt.subplots(4, 2, figsize=(10, 12))
plot_acf(AR1MA0_series, ax=axs[0, 0], lags=20, alpha=0.05)
plot_pacf(AR1MA0_series, ax=axs[0, 1], lags=20, alpha=0.05)
plot_acf(AR0MA1_series, ax=axs[1, 0], lags=20, alpha=0.05)
plot_pacf(AR0MA1_series, ax=axs[1, 1], lags=20, alpha=0.05)
plot_acf(AR1MA1_1_series, ax=axs[2, 0], lags=20, alpha=0.05)
plot_pacf(AR1MA1_1_series, ax=axs[2, 1], lags=20, alpha=0.05)
plot_acf(AR1MA1_2_series, ax=axs[3, 0], lags=20, alpha=0.05)
plot_pacf(AR1MA1_2_series, ax=axs[3, 1], lags=20, alpha=0.05)
axs[0, 0].set_title('AR(1) ACF plot')
axs[0, 1].set_title('AR(1) PACF plot')
axs[1, 0].set_title('MA(1) ACF plot')
axs[1, 1].set_title('MA(1) PACF plot')
axs[2, 0].set_title('ARMA(1,1) ACF plot (1)')
axs[2, 1].set_title('ARMA(1,1) PACF plot (1)')
axs[3, 0].set_title('ARMA(1,1) ACF plot (2)')
axs[3, 1].set_title('ARMA(1,1) PACF plot (2)')
plt.tight_layout()
plt.show()

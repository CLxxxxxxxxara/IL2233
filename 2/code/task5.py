import pandas as pd
import numpy as np
import statsmodels.api as sm
data = pd.read_csv('Temperature.csv')
df = pd.read_csv('Temperature.csv', usecols=[1])
import matplotlib.pyplot as plt

plt.plot(data['year'], data['temperature change'])
plt.xlabel('Year')
plt.ylabel('Temperature change')
plt.title('Global temperature change')
plt.show()

import seaborn as sns

# histogram and density plot
sns.histplot(data['temperature change'], kde=True)
plt.xlabel('Temperature change')
plt.title('Histogram and density plot')
plt.show()

# heatmap and box plot
sns.heatmap(df, annot=True)
plt.title('Heatmap')
plt.show()

sns.boxplot(data['temperature change'])
plt.xlabel('Temperature change')
plt.title('Box plot')
plt.show()

# Calculate IQR
Q1 = data["temperature change"].quantile(0.25)
Q3 = data["temperature change"].quantile(0.75)
IQR = Q3 - Q1

# Identify outliers
outliers = data[(data["temperature change"] < Q1 - 1.5*IQR) | (data["temperature change"] > Q3 + 1.5*IQR)]
print(outliers)

# Lag-1 plot
pd.plotting.lag_plot(data["temperature change"], lag=1)
plt.title("Lag-1 Plot")
plt.show()

# Lag-2 plot
pd.plotting.lag_plot(data["temperature change"], lag=2)
plt.title("Lag-2 Plot")
plt.show()



from statsmodels.stats.diagnostic import acorr_ljungbox

# lbvalue, pvalue =
# sm.stats.acorr_ljungbox(df, lags=10)
# print('Ljung-Box test p-values:', pvalue)
print(sm.stats.acorr_ljungbox(df, lags=[10]))

from statsmodels.tsa.stattools import adfuller

# Perform ADF test on the temperature change data
adf_test = adfuller(df["temperature change"])

# Print the p-value
print(adf_test[1])


# Step 2: Stationarity test and differencing
# visual inspection
# df["Temperature_diff"] = df["temperature change"] - df["temperature change"].shift(1)
# df["Temperature_diff"].dropna().plot(figsize=(12, 8))
df_diff = df.diff().dropna()
df_diff.plot(figsize=(10,6))
plt.title('Differenced Time Series Plot')
plt.show()

# ADF test
adf_result = sm.tsa.stattools.adfuller(df_diff['temperature change'])
print('ADF test p-value:', adf_result[1])

# Step 3: Model identification
# ACF and PACF plots
fig, ax = plt.subplots(2,1,figsize=(10,8))
sm.graphics.tsa.plot_acf(df_diff, lags=30, ax=ax[0])
sm.graphics.tsa.plot_pacf(df_diff, lags=30, ax=ax[1])
plt.show()

# Step 4: Parameter estimation and model optimization
model = sm.tsa.ARIMA(df, order=(1,1,7)).fit()
print(model.summary())

# Step 5: Model validation
# residual plot
model.resid.plot(figsize=(10,6))
plt.title('Residual Plot')
plt.show()

# # Ljung-Box test for residuals
# lbvalue, pvalue = sm.stats.acorr_ljungbox(model.resid, lags=[10])
# print('Ljung-Box test p-values for residuals:', pvalue)
print(sm.stats.acorr_ljungbox(model.resid, lags=[10]))

# MSE for in-sample prediction
y_pred = model.predict()
mse = np.mean((y_pred - df['temperature change'])**2)
print('MSE:', mse)

# Step 6: Model forecasting
# out-of-sample prediction
# Train the ARIMA model on the entire dataset
from statsmodels.tsa.arima.model import ARIMA
# fit ARIMA model to entire dataset
model = ARIMA(df, order=(1, 1, 7))
model_fit = model.fit()
print(model_fit.summary())

# make out-of-sample predictions for next 10 steps
forecast = model_fit.forecast(10)
print(forecast)
oyear = pd.read_csv('Temperature.csv', usecols=[0])
years=[1986,1987,1988,1989,1990,1991,1992,1993,1994,1995]
# plot original data and predicted values
plt.plot(oyear, df.values, label='Original')
plt.plot(years, forecast, label='Predicted')
plt.legend()
plt.show()




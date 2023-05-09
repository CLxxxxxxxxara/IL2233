import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd

q_linear=pd.DataFrame([1,2,3,4,5,6,7])
q_expo=pd.DataFrame([1,2,4,8,16,32,64])

plt.plot(q_linear)
plt.title('linear_original')
plt.show()
df_diff = q_linear.diff().dropna()
df_diff.plot(figsize=(10,6))
plt.title('Differenced linear')
plt.show()

plt.plot(q_expo)
plt.title('expo_original')
plt.show()

log_diff = np.log(q_expo)
log_diff.plot(figsize=(10,6))
plt.title('Transformed expo')
plt.show()

df_diff = log_diff.diff().dropna()
df_diff.plot(figsize=(10,6))
plt.title('Differenced linear')
plt.show()

# Generate a synthetic series with a seasonal trend
n = 200
x = np.arange(n)
y = 10*np.sin(2*np.pi*x/12) + np.random.normal(0, 1, n)

# Plot the original series
plt.plot(x, y)
plt.title('Original series with seasonal trend')
plt.show()

# Plot the autocorrelation function (ACF) to identify the seasonal period
plot_acf(y, lags=50)
plt.title('ACF of original series')
plt.show()

# Apply differencing with step length = 12 to remove the seasonal trend
y_diff = np.diff(y, n=12)

# Plot the differenced series
plt.plot(x[12:], y_diff)
plt.title('Differenced series with removed seasonal trend (step length=12)')
plt.show()

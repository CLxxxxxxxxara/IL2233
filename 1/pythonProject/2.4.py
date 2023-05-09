import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
short_series= [1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(short_series, lags=14, ax=ax1)
plot_pacf(short_series, lags=6, ax=ax2)
plt.show()
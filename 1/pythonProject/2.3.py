import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load data
data = loadmat('03_EEG-1.mat')
EEG = data['EEG'].reshape(-1)
t = data['t'][0]

# Create time vector
time = np.arange(0, len(EEG)/1000, 1/1000)

# Line plot
plt.plot(time, EEG)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (uV)')
plt.title('EEG data')
plt.show()

# Histogram
plt.hist(EEG, bins=50)
plt.xlabel('Voltage (uV)')
plt.ylabel('Count')
plt.title('EEG data')
plt.show()

# Density plot
import seaborn as sns

sns.kdeplot(EEG)
plt.xlabel('EEG Amplitude')
plt.ylabel('Density')
plt.title('EEG Density Plot')
plt.show()


# Box plot
plt.boxplot(EEG)
plt.ylabel('Voltage (uV)')
plt.title('EEG data')
plt.show()

# Lag-1 plot
plt.scatter(EEG[:-1], EEG[1:])
plt.xlabel('EEG(t)')
plt.ylabel('EEG(t+1)')
plt.title('Lag-1 plot')
plt.show()

# ACF plot
plot_acf(EEG, lags=50)
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.title('Autocorrelation function')
plt.show()

# PACF plot

plot_pacf(EEG, lags=50)
plt.xlabel('Lag')
plt.ylabel('PACF')
plt.ylim(-3, 13)
plt.title('Partial autocorrelation function')
plt.show()


# Calculate statistical characteristics
mean_EEG = np.mean(EEG)
var_EEG = np.var(EEG)
std_EEG = np.std(EEG)

# Print the results
print('Mean of EEG data: {:.4f}'.format(mean_EEG))
print('Variance of EEG data: {:.4f}'.format(var_EEG))
print('Standard deviation of EEG data: {:.4f}'.format(std_EEG))

from statsmodels.tsa.stattools import acf

# Compute autocovariance
acov_EEG = acf(EEG, nlags=50, fft=False)

# Plot autocovariance
plt.plot(acov_EEG)
plt.xlabel('Lag')
plt.ylabel('Autocovariance')
plt.title('Autocovariance function')
plt.savefig('autocovariance.png')
plt.show()




# Compute power spectrum using FFT
freq = np.fft.rfftfreq(len(EEG), 1/1000) # Generate frequency axis
spectrum = np.abs(np.fft.rfft(EEG))**2 # Compute power spectrum
spectrum = spectrum[:len(freq)] # Keep only positive frequencies

# Plot power spectrum in linear scale
plt.figure()
plt.plot(freq, spectrum)
plt.xlim([0, 100])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Power spectrum (linear scale)')
plt.show()

# Plot power spectrum in log scale (dB)
plt.figure()
plt.plot(freq, 10*np.log10(spectrum))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB)')
plt.xlim([0, 100])                           # Select the frequency range.
plt.ylim([-60, 0])
plt.title('Power spectrum (log scale)')
plt.show()

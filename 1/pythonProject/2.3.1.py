import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import loadmat                    # To load .mat files
from pylab import *                             # Import plotting functions
from numpy import where
from numpy.fft import fft, rfft
from scipy.signal import spectrogram
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

x = EEG                               # Relabel the data variable
dt = t[1] - t[0]                      # Define the sampling interval
N = x.shape[0]                        # Define the total number of data points
T = N * dt                            # Define the total duration of the data

xf = fft(x - x.mean())                # Compute Fourier transform of x
Sxx = 2 * dt ** 2 / T * (xf * xf.conj())  # Compute spectrum
Sxx = Sxx[:int(len(x) / 2)]           # Ignore negative frequencies

df = 1 / T.max()                      # Determine frequency resolution
fNQ = 1 / dt / 2                      # Determine Nyquist frequency
faxis = arange(0,fNQ,df)              # Construct frequency axis

plot(faxis, Sxx.real)                 # Plot spectrum vs frequency
xlim([0, 100])                        # Select frequency range
xlabel('Frequency [Hz]')              # Label the axes
ylabel('Power [$\mu V^2$/Hz]')
show()

plot(faxis, 10 * log10(Sxx / max(Sxx)))  # Plot the spectrum in decibels.
xlim([0, 100])                           # Select the frequency range.
ylim([-60, 0])                           # Select the decibel range.
xlabel('Frequency [Hz]')                 # Label the axes.
ylabel('Power [dB]')
#savefig('imgs/3-13a')
show()
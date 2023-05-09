import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import statsmodels.api as sm

# Generate the sine wave signals
fs = 200  # sampling rate
t = np.arange(0, 5, 1/fs)  # time vector

frequencies = [10, 20, 30, 40, 50]
signals = []
for freq in frequencies:
    s = np.sin(2*np.pi*freq*t[:fs])  # one second sine wave
    signals.append(s)

# Draw a line plot of the series
plt.figure()
plt.plot(t, np.concatenate(signals))
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Sine wave series')
plt.show()


# Draw  power spectrum (power density graph) (Discrete Fourier Transform) of the series.
# Compute the Discrete Fourier Transform (DFT)
dft = np.fft.fft(np.concatenate(signals))

# Compute the power spectrum
psd = np.abs(dft)**2 / len(dft)
freqs = np.fft.fftfreq(len(dft), 1/fs)

# Plot the power spectrum
plt.figure()
plt.plot(freqs[:len(freqs)//2], psd[:len(psd)//2])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density')
plt.title('Power spectrum of sine wave series (DFT)')
plt.show()

# Draw the spectrogram of the series
f, t_spec, Sxx = signal.spectrogram(np.concatenate(signals), fs)
plt.figure()
plt.pcolormesh(t_spec, f, Sxx)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Spectrogram of sine wave series')
plt.colorbar()
plt.show()


# Draw and compare the ACF and PACF graphs of the first one-second (frequency 10Hz)
# and the second one-second series (frequency 20Hz), with lags up to 50
s_10Hz = signals[0]
s_20Hz = signals[1]

# ACF and PACF of 10Hz signal
lags = np.arange(0, 51)
acf_10Hz = sm.tsa.acf(s_10Hz, nlags=50)
pacf_10Hz = sm.tsa.pacf(s_10Hz, nlags=50)

# ACF and PACF of 20Hz signal
acf_20Hz = sm.tsa.acf(s_20Hz, nlags=50)
pacf_20Hz = sm.tsa.pacf(s_20Hz, nlags=50)

plt.figure()
plt.subplot(2, 2, 1)
plt.stem(lags, acf_10Hz)
plt.title('ACF (10Hz)')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.subplot(2, 2, 2)
plt.stem(lags, pacf_10Hz)
plt.title('PACF (10Hz)')
plt.xlabel('Lag')
plt.ylabel('Partial autocorrelation')
plt.subplot(2, 2, 3)
plt.stem(lags, acf_20Hz)
plt.title('ACF (20Hz)')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.subplot(2, 2, 4)
plt.stem(lags, pacf_20Hz)
plt.title('PACF (20Hz)')
plt.xlabel('Lag')
plt.ylabel('Partial autocorrelation')
plt.tight_layout()
plt.show()

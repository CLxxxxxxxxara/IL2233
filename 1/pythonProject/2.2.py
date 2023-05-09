import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch

# Load the data
data = loadmat('02_EEG-1.mat')
EEGa = data['EEGa']
EEGb = data['EEGb']
t = data['t'][0]

# Calculate the ERP for each condition
ERP_a = np.mean(EEGa, axis=0)
ERP_b = np.mean(EEGb, axis=0)

# Plot the ERP for both conditions
plt.plot(t, ERP_a, label='Condition A')
plt.plot(t, ERP_b, label='Condition B')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (uV)')
plt.title('Event-Related Potentials')
plt.legend()
plt.show()


# Plot the PSD for Condition A
signals_a = np.concatenate(EEGa)


dft_a = np.fft.fft(signals_a)
psd_a = np.abs(dft_a)**2 / len(dft_a)
freqs_a = np.fft.fftfreq(len(dft_a), 1/500)


plt.figure()
plt.plot(freqs_a[:len(freqs_a)//2], psd_a[:len(psd_a)//2])
plt.xlabel('Frequency (Hz)')
plt.xlim(0,20)
plt.ylabel('Power spectral density')
plt.title('Power spectrum of Condition A')
plt.show()
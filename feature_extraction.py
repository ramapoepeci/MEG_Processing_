import numpy as np
from scipy.signal import welch

def calculate_psd(epochs, sfreq, nperseg):
    data = epochs.get_data(copy=False)
    psds = []
    for epoch in data:
        freqs, psd = welch(epoch, sfreq, nperseg=nperseg)
        psds.append(psd)
    psds = np.array(psds)
    return freqs, psds

def extract_features(freqs, psds):
    freq_60_idx = np.argmin(np.abs(freqs - 60))
    power_60_hz = psds[:, :, freq_60_idx]
    return power_60_hz

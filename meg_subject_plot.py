import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt

def plot_subject_data(subject_id, data_folder):
    # Construct the file path for the raw FIF file and event file

    #CHANGE FILE PATH
    raw_file_path = f'change file path'
    # Load the raw data
    raw = mne.io.read_raw_fif(raw_file_path, preload=True)
    raw.interpolate_bads()
    # Find events
    events = mne.find_events(raw, stim_channel="STI 014")

    # Compute and plot PSD
    psd_fig = raw.compute_psd(tmax=np.inf).plot(
        average=True, picks="data", exclude="bads"
    )

    # Arrow annotations 
    for ax in psd_fig.axes[1:]:
        freqs = ax.lines[-1].get_xdata()
        psds = ax.lines[-1].get_ydata()
        for freq in (60, 120, 180, 240):
            idx = np.searchsorted(freqs, freq)
            ax.arrow(
                x=freqs[idx],
                y=psds[idx] + 18,
                dx=0,
                dy=-12,
                color="red",
                width=0.1,
                head_width=3,
                length_includes_head=True,
            )

    # Plot the raw data
    raw.plot(duration=5, n_channels=30)
    plt.show()

subject_ids = ['sub-004', 'sub-007', 'sub-008', 'sub-009', 'sub-012']
#CHANGE FILE PATH
base_data_folder = 'change file path'

for subject_id in subject_ids:
    plot_subject_data(subject_id, base_data_folder)

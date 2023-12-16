import mne
#Load each raw file
def load_raw_data(filepath):
    raw = mne.io.read_raw_fif(filepath, preload=True)
    return raw



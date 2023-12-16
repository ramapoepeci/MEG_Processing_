import mne

# Define the time interval for epochs (200 ms before to 500 ms after the event)
tmin = -0.2
tmax = 0.5

def create_epochs(raw, events, event_id, tmin, tmax):
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(None, 0), preload=True)
    return epochs

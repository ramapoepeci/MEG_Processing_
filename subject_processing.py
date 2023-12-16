from process import load_raw_data
from event_handling import read_event_file, create_mne_events, create_event_id_dict
from create_epochs import create_epochs
from feature_extraction import calculate_psd, extract_features
from labeling import get_labels
import matplotlib.pyplot as plt

def process_single_subject(subject_id, base_dir):
    raw_file_path = f'{base_dir}/data/sing_aud/{subject_id}_task-words_meg.fif'
    event_file_path = f'{base_dir}/data/sing_aud/{subject_id}_task-words_events.tsv'

    # Load raw data and process events
    raw = load_raw_data(raw_file_path)
    events_df = read_event_file(event_file_path)
    mne_events = create_mne_events(events_df)
    event_id = create_event_id_dict(events_df)

    # Create epochs
    epochs = create_epochs(raw, mne_events, event_id, tmin=-0.2, tmax=0.5)

    # Calculate PSD and extract features
    sfreq = epochs.info['sfreq']
    nperseg = min(int(round(sfreq * 1.0)), epochs.get_data(copy=False).shape[2])
    freqs, psds = calculate_psd(epochs, sfreq, nperseg)
    features = extract_features(freqs, psds)

    # Obtain labels for each epoch
    labels = get_labels(epochs, event_id)

    return features, labels, epochs


def plot_epochs_psd(epochs, fmin=1, fmax=100, bandwidth=1.5, subject_id='unknown'):
    # Plot the PSD for the provided epochs
    psd_fig = epochs.plot_psd(fmin=fmin, fmax=fmax, bandwidth=bandwidth, spatial_colors=False)
    psd_fig.suptitle(f'Subject {subject_id} PSD')
    plt.show()
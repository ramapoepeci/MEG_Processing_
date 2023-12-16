import pandas as pd
import numpy as np
import mne

def read_event_file(event_file_path):
    events_df = pd.read_csv(event_file_path, sep='\t')
    return events_df

def create_mne_events(events_df):
    mne_events = np.column_stack((events_df['sample'].values.astype(int), np.zeros(len(events_df), dtype=int), events_df['value'].values.astype(int)))
    return mne_events

def create_event_id_dict(events_df):
    event_id = {str(evt): val for evt, val in zip(events_df['trial_type'].unique(), events_df['value'].unique())}
    return event_id

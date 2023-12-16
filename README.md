In order for the code to work, the full dataset must be downloaded and extracted from the github page, https://github.com/OpenNeuroDatasets/ds004276.git.
Once downloaded, take the sub-000_task-words_events.tsv and sub-000_task-words_meg.fif file from each from each sub-ooo/meg/ folder and place them in a new a folder called data in the same directory of the main code.

**change file paths and directoreies**

**in main.py**
line 21: change_direc = 'C:/python tt/eegproject'  Change to your specified directory
line 32: raw_file_path = f'C:/python tt/eegproject/data/sing_aud/{subject_id}_task-words_meg.fif' Change to file path that leads to .fif files
line 33: event_file_path = f'C:/python tt/eegproject/data/sing_aud/{subject_id}_task-words_events.tsv' Change to file path that leads to .tsv files

**In plot_subject_data.py**
Line 10: raw_file_path = f'C:/python tt/eegproject/data/sing_aud/{subject_id}_task-words_meg.fif' Change to file path that leads to .fif files
Line 45: base_data_folder = 'c:/Users/eran/Desktop/vep_projects/sing_aud/audi_sing_word_meg' Change to file path that leads to .fif files


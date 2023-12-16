import os
import numpy as np
from process import load_raw_data
from event_handling import read_event_file, create_mne_events, create_event_id_dict
from create_epochs import create_epochs
from feature_extraction import calculate_psd, extract_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from labeling import get_labels
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from subject_processing import process_single_subject, plot_epochs_psd
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Change directory
change_direc = 'C:/python tt/eegproject'
os.chdir(change_direc)


def process_subjects(subject_ids):
    all_features = []
    all_labels = []
    event_id_mappings = {} 
    
    for subject_id in subject_ids:
        #CHANGE FILE PATH
        raw_file_path = f'CHANGE FILE PATH/data/{subject_id}_task-words_meg.fif'
        event_file_path = f'CHANGE FILE PATH/data/{subject_id}_task-words_events.tsv'
        X, y, epochs = process_single_subject(subject_id, change_direc)  
        # Load raw data and process events
        raw = load_raw_data(raw_file_path)
        events_df = read_event_file(event_file_path)
        mne_events = create_mne_events(events_df)
        event_id = create_event_id_dict(events_df)
        event_id_mappings[subject_id] = event_id
        # Create epochs
        epochs = create_epochs(raw, mne_events, event_id, tmin=-0.2, tmax=0.5)

        # Calculate PSD and extract features
        sfreq = epochs.info['sfreq']
        nperseg = min(int(round(sfreq * 1.0)), epochs.get_data(copy=False).shape[2])
        freqs, psds = calculate_psd(epochs, sfreq, nperseg)
        power_60_hz = extract_features(freqs, psds)

        # Obtain labels for each epoch
        labels = get_labels(epochs, event_id)

        # Append features and labels to the combined list
        all_features.append(power_60_hz)
        all_labels.append(labels)

    # Combine features and labels from all subjects
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0), event_id_mappings


all_subjects = ['sub-004', 'sub-007', 'sub-008', 'sub-009', 'sub-012']

# Loop over each subject
for subject_id in all_subjects:
    X, y, epochs = process_single_subject(subject_id, change_direc)
    # Plot PSD
    plot_epochs_psd(epochs, fmin=1, fmax=100, bandwidth=1.5, subject_id=subject_id)

train_subjects = ['sub-004', 'sub-007', 'sub-008', 'sub-009', 'sub-012']
test_subjects = ['sub-002', 'sub-003', 'sub-005', 'sub-013', 'sub-014', 'sub-015', 'sub-018']

# Process training 
X_train, y_train, train_event_id_mappings = process_subjects(train_subjects)
X_test, y_test, test_event_id_mappings = process_subjects(test_subjects)

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


#Random Forest Classification with cross validation
all_subjects = ['sub-004', 'sub-007', 'sub-008', 'sub-009', 'sub-012', 'sub-002', 'sub-003', 'sub-005', 'sub-013', 'sub-014', 'sub-015', 'sub-018']

X, y, event_id_mappings = process_subjects(all_subjects)
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

cv_strategy = StratifiedKFold(n_splits=5)
cross_val_scores = cross_val_score(clf, X, y, cv=cv_strategy, scoring='accuracy')
print(f"Cross-Validation Scores: {cross_val_scores}")
print(f"Mean CV Accuracy: {np.mean(cross_val_scores)}")

#train
clf.fit(X_train, y_train)
feature_importances = clf.feature_importances_
plt.bar(range(len(feature_importances)), feature_importances)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()


#TSNE Plotting
X_embedded = TSNE(n_components=2).fit_transform(X_train)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_train)
plt.colorbar()
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.title('t-SNE Visualization of Training Data')
plt.show()



cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

y_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_bin.shape[1]

# Compute ROC curve and ROC area for each class
clf_probs = clf.predict_proba(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], clf_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotting For Flase Postive
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for Class {i}')
    plt.legend(loc="lower right")
    plt.show()




#Determine Labelling of Epoch
def get_labels(epochs, event_id):
    labels = []
    for epoch in epochs.events:
        if epoch[2] == event_id['item']:
            labels.append(0)  # Label for 'item'
        elif epoch[2] == event_id['no_probe']:
            labels.append(1)  # Label for 'no_probe'
        elif epoch[2] == event_id['item_post_probe']:
            labels.append(2)  # Label for 'item_post_probe'
        elif epoch[2] == event_id['yes_probe']:
            labels.append(3)  # Label for 'yes_probe'
        else:
            labels.append(4)  # Default label for other events
    return labels


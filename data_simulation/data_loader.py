import numpy as np
import os
from sklearn.model_selection import train_test_split

from susceptibility_model_data import *
from obs_to_tensor import obs_to_array

# Assuming you have the functions load_data, obs_to_array, and load_labels defined
def load_all_data(file_number, max_labels=5):
    fname_topo = f"data_simulation/outputs/magnetics_topo_{file_number}.txt"
    fname_data = f"data_simulation/outputs/magnetics_data_{file_number}.obs"
    fname_labels = f"data_simulation/sphere_data_labels/sphere_label_{file_number}.npy"
    
    obs, receiver_locations, _ = load_data(fname_topo, fname_data)
    data_array = obs_to_array(obs, receiver_locations)
    labels = load_labels(fname_labels)
    
    # Add the special marker to the end of the labels
    special_marker = np.array([999, 999, 999, 999])  # Customize this as per your needs
    
    # Pad the labels to have a consistent shape (max_labels x 4)
    padded_labels = np.full((max_labels, 4), special_marker, dtype=np.float32)
    padded_labels[:labels.shape[0], :] = labels

    return data_array, padded_labels

# Example usage
num_files = 200
all_data = []
all_labels = []

# Specify the maximum number of labels to pad to
max_labels = 5

for file_number in range(num_files):
    data_array, labels = load_all_data(file_number, max_labels=max_labels)
    all_data.append(data_array)
    all_labels.append(labels)

# Convert to numpy arrays
all_data = np.array(all_data)
all_labels = np.array(all_labels)

# Save all_data
np.save("data_simulation/processed_data/all_data.npy", all_data)

# Load all_data (if needed)
# all_data = np.load("all_data.npy")

# Split into train and test sets
train_data, test_data, train_labels, test_labels = train_test_split(all_data, all_labels, test_size=0.15, random_state=42)

# Optionally, you can save the train and test sets
np.save("data_simulation/processed_data/train_data.npy", train_data)
np.save("data_simulation/processed_data/test_data.npy", test_data)
np.save("data_simulation/processed_data/train_labels.npy", train_labels)
np.save("data_simulation/processed_data/test_labels.npy", test_labels)

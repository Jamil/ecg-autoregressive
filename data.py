import numpy as np
import random
import scipy.io
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    ecg_mat = scipy.io.loadmat(path)
    ecg_data = ecg_mat['ECGData'][0][0][0]
    labels = ecg_mat['ECGData'][0][0][1]
    return ecg_data, labels

def split_dataset_indices(N, train_percent, val_percent, test_percent, seed=42):
    # Validate input percentages
    if train_percent + val_percent + test_percent != 100:
        raise ValueError("The sum of the percentages must be 100.")

    random.seed(seed)
    # Generate a list of indices
    indices = list(range(N))
    random.shuffle(indices)

    # Calculate split sizes
    train_size = int((train_percent / 100) * N)
    val_size = int((val_percent / 100) * N)

    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    return train_indices, val_indices, test_indices

def create_windowed_dataset(data, labels, window_size=1024, stride=None):
    if not stride:
        stride = window_size

    min_val = np.min(data, axis=(0, 1), keepdims=True)
    max_val = np.max(data, axis=(0, 1), keepdims=True)
    data = 2 * ((data - min_val) / (max_val - min_val)) - 1

    num_samples, sample_size = data.shape
    num_windows = sample_size // window_size

    # Initialize the windowed data and label arrays
    windowed_data = np.empty((num_samples * num_windows, window_size))
    windowed_labels = np.empty((num_samples * num_windows, 1), dtype=labels.dtype)

    seqs = []
    for i in range(num_samples):
        for j in range(num_windows):
            start = j * stride
            end = start + window_size
            windowed_data[i * num_windows + j, :] = data[i, start:end]
            windowed_labels[i * num_windows + j, :] = labels[i]
        seqs.append(num_windows)
    seq_lens = np.array(seqs)

    labels_flattened = np.array([label[0][0] for label in windowed_labels])

    # Encode the string labels to integers
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels_flattened)

    return windowed_data, labels_encoded, seq_lens

def generate_sequences(embeddings, sequence_lengths, window_size=10):
    features, labels = [], []

    start_idx = 0
    for length in sequence_lengths:
        for i in range(start_idx, start_idx + length - window_size):
            features.append(embeddings[i:i + window_size])
            labels.append(embeddings[i + window_size])
        start_idx += length

    return np.array(features), np.array(labels)

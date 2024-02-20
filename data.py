import h5py
import numpy as np
import os
import pandas as pd
import random
import scipy.io
import tensorflow as tf
import torch

from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

H5_PATH = '/home/ec2-user/ED_waveform/troponin_ecg.h5'
MAT_PATH = '/home/ec2-user/ecg-autoregressive/ECGData.mat'

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, patch_size=64, cache_dir='cache'):
        self.file_list = file_list
        self.patch_size = patch_size
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _cache_file_path(self, idx):
        base_name = os.path.basename(self.file_list[idx])
        return os.path.join(self.cache_dir, f"{base_name}_cache_{self.patch_size}.npz")

    def __len__(self):
        return len(self.file_list) 

    def __getitem__(self, idx):
        try:
            cache_file = self._cache_file_path(idx)
            if os.path.exists(cache_file):
                # Load from cache
                with np.load(cache_file) as data:
                    x_seq = data['x_seq']
                    y_seq = data['y_seq']
            else:
                # Process and cache
                file_path = self.file_list[idx]
                data = pd.read_csv(file_path)
                flattened_data = data.values.flatten()
                ecg_windows = create_windowed_dataset(flattened_data, window_size=self.patch_size)
                x_seq, y_seq = generate_sequences(ecg_windows)
                np.savez(cache_file, x_seq=x_seq, y_seq=y_seq)
            return torch.tensor(x_seq, dtype=torch.float), torch.tensor(y_seq, dtype=torch.float)
        except ValueError as e:
            print(f"Error loading file {cache_file}: {e}")
            # Handle the error: you could choose to skip this sample, return a dummy sample, etc.
            # Here's an example of returning a dummy sample:
            return torch.zeros((1, 10, self.patch_size,)), torch.zeros((1, self.patch_size,))

class BIHDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, window_size, transform=None, autoregressive=False):
        self.data_path = data_path
        self.window_size = window_size
        self.transform = transform
        self.autoregressive = autoregressive
        self.ecg_data, self.raw_labels = self.load_data(data_path)
        self.label_encoder = LabelEncoder()
        self.labels = self.encode_labels(self.raw_labels)
        self.x_seq, self.y_seq = self.prepare_sequences(self.ecg_data)

    def load_data(self, path):
        ecg_mat = scipy.io.loadmat(path)
        ecg_data = ecg_mat['ECGData'][0][0][0]
        labels = ecg_mat['ECGData'][0][0][1]
        return ecg_data, labels

    def prepare_sequences(self, ecg_data):
        x_seqs, y_seqs = [], []
        for i in range(ecg_data.shape[0]):
            windows = create_windowed_dataset(ecg_data[i], self.window_size)

            x_seq, y_seq = generate_sequences(windows)
            x_seqs.append(x_seq)
            if self.autoregressive:
                y_seqs.append(y_seq)
            else:
                y_seqs.append(self.labels[i])
        x_seqs = np.array(x_seqs)
        y_seqs = np.array(y_seqs)
        return x_seqs, y_seqs

    def encode_labels(self, raw_labels):
        # Flatten the label array and extract strings
        labels = [label[0][0] for label in raw_labels]

        # Encode labels
        self.label_encoder.fit(labels)
        encoded_labels = self.label_encoder.transform(labels)

        return encoded_labels

    def __len__(self):
        return len(self.x_seq)

    def __getitem__(self, idx):
        x = self.x_seq[idx]
        y = self.y_seq[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

class HDF5Dataset(Dataset):
    def __init__(self, h5_path, split_type='train', split_ratio=0.8, window_size=64, random_state=42, autoregressive=False, embedder=None):
        self.h5_path = h5_path
        self.window_size = window_size
        self.split_type = split_type
        self.autoregressive = autoregressive
        self.embedder = embedder
        # Load group keys and split
        with h5py.File(self.h5_path, 'r') as f:
            all_keys = list(f.keys())
        train_keys, test_keys = train_test_split(all_keys, train_size=split_ratio, random_state=random_state)
        # Further split test set into validation and test sets
        val_keys, test_keys = train_test_split(test_keys, train_size=0.5, random_state=random_state)
        if split_type == 'train':
            self.group_keys = train_keys
        elif split_type == 'val':
            self.group_keys = val_keys
        elif split_type == 'test':
            self.group_keys = test_keys
        else:
            raise ValueError(f"Unsupported split_type: {split_type}")

    def __len__(self):
        return len(self.group_keys)

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            group = f[self.group_keys[idx]]
            ecg_data = group['ECG'][:]
            label = group['Label'][()]
            windowed_ecg = create_windowed_dataset(ecg_data, self.window_size)
            x_seq, y_seq = generate_sequences(windowed_ecg)

            if not self.autoregressive:
                y_seq = np.full(len(x_seq), label)

            if self.embedder:
                x_seq = self.embedder.embed(x_seq)

        return torch.tensor(x_seq, dtype=torch.float), torch.tensor(y_seq, dtype=torch.float)

class HDF5DatasetKeras(Sequence):
    def __init__(self, h5_path, batch_size=32, split_type='train', split_ratio=0.8, window_size=64, random_state=42, autoregressive=False, embedder=None, subsample=None):
        self.h5_path = h5_path
        self.window_size = window_size
        self.split_type = split_type
        self.batch_size = batch_size
        self.autoregressive = autoregressive
        self.embedder = embedder
        self.subsample = subsample
        
        # Load group keys and split
        with h5py.File(self.h5_path, 'r') as f:
            all_keys = list(f.keys())
        train_keys, test_keys = train_test_split(all_keys, train_size=split_ratio, random_state=random_state)
        
        # Further split test set into validation and test sets
        val_keys, test_keys = train_test_split(test_keys, train_size=0.5, random_state=random_state)
        if split_type == 'train':
            self.group_keys = train_keys
        elif split_type == 'val':
            self.group_keys = val_keys
        elif split_type == 'test':
            self.group_keys = test_keys
        else:
            raise ValueError(f"Unsupported split_type: {split_type}")

    def __len__(self):
        # Calculate the number of batches per epoch
        return np.ceil(len(self.group_keys) / self.batch_size).astype(int)

    def __getitem__(self, idx):
        # Determine the indices of the data for this batch
        batch_keys = self.group_keys[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_x = []
        batch_y = []
        
        with h5py.File(self.h5_path, 'r') as f:
            for key in batch_keys:
                group = f[key]
                ecg_data = group['ECG'][:]
                label = group['Label'][()]
                windowed_ecg = create_windowed_dataset(ecg_data, self.window_size)
                x_seq, y_seq = generate_sequences(windowed_ecg)

                if not self.autoregressive:
                    y_seq = np.full(len(x_seq), label)

                if self.embedder:
                    x_seq = self.embedder.embed(x_seq)

                batch_x.extend(x_seq)
                batch_y.extend(y_seq)

            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)

        if self.subsample is not None and len(x_seq) > self.subsample:
            indices = np.arange(len(x_seq))
            sampled_indices = np.random.choice(indices, self.subsample, replace=False)

            batch_x = batch_x[sampled_indices]
            batch_y = batch_y[sampled_indices]
        
        return batch_x, batch_y

def get_splits(patient_data_path):
    complete_file_list = [os.path.join(patient_data_path, subdir, file) for subdir in os.listdir(patient_data_path)
                      if os.path.isdir(os.path.join(patient_data_path, subdir))
                      for file in os.listdir(os.path.join(patient_data_path, subdir))
                      if file.endswith('.csv')]
    return split_file_list(complete_file_list)

def split_file_list(file_list, train_ratio=0.7, val_ratio=0.15):
    random.seed(42)
    random.shuffle(file_list)
    n = len(file_list)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return file_list[:train_end], file_list[train_end:val_end], file_list[val_end:]

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

def create_windowed_dataset(data, window_size=1024, stride=None):
    if not stride:
        stride = window_size

    # Normalize the data if needed
    min_val = np.min(data)
    max_val = np.max(data)
    data = 2 * ((data - min_val) / (max_val - min_val)) - 1

    # Calculate the number of windows that can be created
    num_windows = (len(data) - window_size) // stride + 1

    # Initialize the windowed data array
    windowed_data = np.empty((num_windows, window_size))

    # Create windows
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        windowed_data[i, :] = data[start:end]

    return windowed_data

def generate_sequences(windowed_data, window_size=10):
    num_windows = windowed_data.shape[0]
    features, labels = [], []

    # Iterate over the windowed data to create sequences
    for i in range(num_windows - window_size):
        features.append(windowed_data[i:i + window_size])
        labels.append(windowed_data[i + window_size])

    return np.array(features), np.array(labels)


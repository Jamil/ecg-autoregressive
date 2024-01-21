import argparse
import logging
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from data import *
from model import CombinedModel

def main():
    set_up_logging()
    parser = set_up_parser()
    args = parser.parse_args()

    window_size = args.window_size
    latent_dim = args.latent_dim

    data, labels = load_data(args.data_path)
    train_indices, val_indices, test_indices = split_dataset_indices(labels.shape[0], 70, 15, 15)

    ecg_t, labels_t, seq_t = create_windowed_dataset(
        data[train_indices],
        labels[train_indices],
        window_size=window_size
    )

    ecg_v, labels_v, seq_v = create_windowed_dataset(
        data[val_indices],
        labels[val_indices],
        window_size=window_size
    )

    ecg_tx_seq, ecg_ty_seq = generate_sequences(ecg_t, seq_t, window_size=10)
    ecg_vx_seq, ecg_vy_seq = generate_sequences(ecg_v, seq_v, window_size=10)

    # Convert numpy arrays to PyTorch tensors
    train_dataset = TensorDataset(torch.Tensor(ecg_tx_seq), torch.Tensor(ecg_ty_seq))
    val_dataset = TensorDataset(torch.Tensor(ecg_vx_seq), torch.Tensor(ecg_vy_seq))

    # Create PyTorch DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    combined = CombinedModel(window_size, latent_dim, args.transformer_layers).to(device)
    optimizer = optim.Adam(combined.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        combined.train()
        train_loss = 0
        with tqdm(total=len(train_loader.dataset), desc=f'Epoch {epoch + 1}/{args.epochs}', unit='sample') as pbar:
            for ecg, labels in train_loader:
                ecg, labels = ecg.to(device), labels.to(device)
                optimizer.zero_grad()
                output = combined(ecg)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * ecg.size(0)
                pbar.set_postfix({'train_loss': loss.item()})
                pbar.update(ecg.size(0))

        # Validation loop
        combined.eval()
        val_loss = 0
        with torch.no_grad():
            for ecg, labels in val_loader:
                ecg, labels = ecg.to(device), labels.to(device)
                output = combined(ecg)
                val_loss += criterion(output, labels).item() * ecg.size(0)

        # Calculate average losses
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{args.epochs} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

    # After training
    ecg_ty_seq_pred = predict(combined, train_dataset, args.batch_size, device)
    ecg_vy_seq_pred = predict(combined, val_dataset, args.batch_size, device)

    plot_examples(args.output_dir, ecg_vy_seq, ecg_vy_seq_pred, window_size)

def set_up_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_up_parser():
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser(description='ECG Autoencoder Training Script')
    parser.add_argument('-d', '--data-path', type=str, required=True, help='Path to the ECG dataset')
    parser.add_argument('-o', '--output-dir', type=str, required=True, help='Directory to save output')
    parser.add_argument('-w', '--window-size', type=int, default=64, help='Window size for the model')
    parser.add_argument('-l', '--latent-dim', type=int, default=16, help='Latent dimension size for the model')
    parser.add_argument('-t', '--transformer-layers', type=int, default=1, help='Number of transformer block layers')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--device', type=str, default=default_device)
    return parser

def plot_examples(output_dir, ecg_seq, ecg_seq_pred, window_size):
    # Plot and save a single sample
    idx = 1917
    plot_sample(ecg_seq[idx], ecg_seq_pred[idx], idx, window_size, f'{output_dir}/sample_plot.png')

    # Plot and save a sequence
    plot_sequence(ecg_seq, ecg_seq_pred, 50, 75, f'{output_dir}/sequence_plot.png')

def predict(model, dataset, batch_size, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for ecg, _ in DataLoader(dataset, batch_size=batch_size):
            ecg = ecg.to(device)
            preds = model(ecg)
            predictions.append(preds.cpu().numpy())
    return np.concatenate(predictions, axis=0)

def plot_sample(actual, pred, idx, window_size, filename):
    plt.plot(actual, 'b')
    plt.plot(pred, 'r')
    plt.fill_between(np.arange(window_size), actual, pred, color='lightcoral')
    plt.legend(labels=['Input', 'Reconstruction', 'Error'])
    plt.savefig(filename)
    plt.close()

def plot_sequence(actuals, preds, start, stop, filename):
    actual = np.concatenate(actuals[start:stop])
    pred = np.concatenate(preds[start:stop])
    plt.plot(actual, 'b')
    plt.plot(pred, 'r')
    plt.fill_between(np.arange((stop - start) * len(actuals[0])), actual, pred, color='lightcoral')
    plt.legend(labels=['Input', 'Reconstruction', 'Error'])
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    main()

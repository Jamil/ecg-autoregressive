import argparse
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from vit import *
from data import *
from model import CombinedModel, TEModel

def main():
    parser = set_up_parser()
    args = parser.parse_args()

    output_dir, hyperparams = create_output_directory(args)
    set_up_logging(args, output_dir, hyperparams)

    window_size = args.window_size
    latent_dim = args.latent_dim

    # Create dataset instances for each split
    train_dataset = HDF5Dataset(H5_PATH, window_size=args.window_size, split_type='train', autoregressive=True)
    val_dataset = HDF5Dataset(H5_PATH, window_size=args.window_size, split_type='val', autoregressive=True)
    test_dataset = HDF5Dataset(H5_PATH, window_size=args.window_size, split_type='test', autoregressive=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#   combined = TEModel(
#           window_size, 
#           latent_dim, 
#           num_transformer_blocks = args.num_transformer_blocks,
#           encoder_hidden_dim = args.encoder_hidden_dim,
#           dim_feedforward = args.dim_feedforward,
#           num_heads = args.num_heads
#   ).to(device)
    combined = vit_base(patch_size=window_size).to(device)
    optimizer = optim.AdamW(combined.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        combined.train()
        train_loss = 0
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}/{args.epochs}', unit='sample') as pbar:
            for i, (ecg, labels) in enumerate(train_dataset):
                n, seq_len, patch_size = ecg.shape
                ecg, labels = ecg.float().to(device), labels.float().to(device)
                optimizer.zero_grad()
                output = combined(ecg)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * ecg.size(0)
                pbar.update(1)

        # Validation loop
        combined.eval()
        val_loss = 0
        with torch.no_grad():
            for ecg, labels in val_dataset:
                ecg, labels = ecg.float().to(device), labels.float().to(device)
                output = combined(ecg)
                val_loss += criterion(output, labels).item() * ecg.size(0)

        # Calculate average losses
        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        if args.wandb:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        print(f"Epoch {epoch + 1}/{args.epochs} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

    # After training
    torch.save(combined.state_dict(), os.path.join(output_dir, 'model.pth'))

    val_samples_limit = 100
    with torch.no_grad():
        for i, (ecg, labels) in enumerate(val_dataset):
            ecg, labels = ecg.float().to(device), labels.float().to(device)
            output = combined(ecg).cpu()
            ecg, labels = ecg.cpu(), labels.cpu()
            plot(output_dir, i, ecg, labels, output, index=0, enable_wandb=args.wandb)
            if i > val_samples_limit:
                break

def set_up_logging(args, output_dir, hyperparams):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if args.wandb:
        wandb.init(
            project='ecg-autoregressive',
            entity='dklab-ed-modeling',
            name=output_dir,
            config=hyperparams
        )

def create_output_directory(args):
    # Generate directory name based on hyperparameters
#   dir_name = f"models/ws{args.window_size}_ld{args.latent_dim}_tl{args.num_transformer_blocks}_e{args.epochs}_bs{args.batch_size}_ed{args.encoder_hidden_dim}_df{args.dim_feedforward}_h{args.num_heads}"
    dir_name = 'vit_base'

    model_summary = f'''
    Autoregressive task: (10, {args.window_size}) -> ({args.window_size},)
    (10, {args.window_size}) -> each patch encoded with ({args.window_size},) -> dense ({args.window_size}, {args.encoder_hidden_dim}) -> dense ({args.encoder_hidden_dim}, {args.latent_dim}) -> ({args.latent_dim})
    Patches of (10, {args.latent_dim}) into {args.num_transformer_blocks}-layer transformer with {args.num_heads} heads and feedforward dim {args.dim_feedforward}, to produce
    (10, {args.latent_dim}) -> dense (10 * {args.latent_dim}, {args.dim_feedforward}) -> dense ({args.dim_feedforward}, {args.latent_dim}) -> ({args.latent_dim})
    Then decoded,
    ({args.latent_dim}) -> dense ({args.latent_dim}, {args.encoder_hidden_dim}) -> dense ({args.encoder_hidden_dim}, {args.window_size}) -> ({args.window_size})
    which is the reconstructed patch.
    '''

    print(model_summary)

    # Serialize hyperparameters into a JSON format
    hyperparams = {
        'window_size': args.window_size,
        'latent_dim': args.latent_dim,
        'num_transformer_blocks': args.num_transformer_blocks,
        'dim_feedforward': args.dim_feedforward,
        'encoder_hidden_dim': args.encoder_hidden_dim,
        'num_heads': args.num_heads,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'device': args.device
    }

    # Check if directory exists, if not, create it
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Directory {dir_name} created.")
    else:
        print(f"Directory {dir_name} already exists.")

    # Save the hyperparameters to a JSON file within the output directory
    json_path = os.path.join(dir_name, 'hyperparameters.json')
    with open(json_path, 'w') as json_file:
        json.dump(hyperparams, json_file, indent=4)
        print(f"Hyperparameters saved to {json_path}.")

    return dir_name, hyperparams

def set_up_parser():
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser(description='ECG Autoencoder Training Script')
    parser.add_argument('-d', '--data-path', type=str, help='Path to the ECG dataset', default=H5_PATH)
    parser.add_argument('-w', '--window-size', type=int, default=64, help='Window size for the model')
    parser.add_argument('-l', '--latent-dim', type=int, default=16, help='Latent dimension size for the model')
    parser.add_argument('-t', '--num-transformer-blocks', type=int, default=1, help='Number of transformer block layers')
    parser.add_argument('-ed', '--encoder-hidden-dim', type=int, default=64, help='Size of hidden dimension in encoder')
    parser.add_argument('-df', '--dim-feedforward', type=int, default=256, help='Feedforward dimension of the transformer')
    parser.add_argument('-nh', '--num-heads', type=int, default=4, help='Number of heads of the transformer')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--device', type=str, default=default_device)
    parser.add_argument('--wandb', action='store_true', help='Enable tracking in wandb')
    return parser

def plot(output_dir, dataset_index, ecg, labels, output, index=0, enable_wandb=False):
    first_ecg = ecg[index]      # Shape: [seq_len, patch_size]
    first_labels = labels[index]  # Shape: [patch_size]
    first_output = output[index]  # Shape: [patch_size]

    # Flatten the first_ecg to reconstruct the original signal for the first sequence
    original_signal = first_ecg.flatten()

    # Create time steps for the original signal and the prediction
    time_steps_original = np.arange(original_signal.shape[0])
    time_steps_prediction = np.arange(original_signal.shape[0], original_signal.shape[0] + first_labels.shape[0])

    # Plotting
    plt.figure(figsize=(15, 5))
    plt.plot(time_steps_original, original_signal, label='Original Signal')
    plt.plot(time_steps_prediction, first_labels, label='Labels', linestyle='--')
    plt.plot(time_steps_prediction, first_output, label='Output', linestyle='-.')
    plt.xlabel('Time Steps')
    plt.ylabel('Signal Amplitude')
    plt.title(f'ECG Signal @ {dataset_index} with Labels and Predicted Output Overlay')
    plt.legend()

    image_dir = os.path.join(output_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)

    image_save_path = os.path.join(image_dir, f'{dataset_index}-{index}.png')
    plt.savefig(image_save_path)
    plt.close()
                
def predict(model, dataset, index, device):
    model.eval()
    with torch.no_grad():
        ecg, _ = dataset[index]
        ecg = ecg.float().to(device)
        preds = model(ecg)
        return preds.cpu().numpy()

if __name__ == '__main__':
    main()

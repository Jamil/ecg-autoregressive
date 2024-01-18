## Autoregressive ECG training script documentation

### Overview
This script trains an autoregressive model on ECG data. It handles data loading, preprocessing, model training, validation, making predictions, and visualizations.

### Key components
- **Data Preprocessing**: Creates windowed datasets and sequences for training.
- **Model**: Uses `CombinedModel` from `model.py`, an autoregressive model for ECG data.
- **Training and Validation**: Trains and validates the model with specified hyperparameters.
- **Visualization**: Plots input data and model reconstructions.

### Usage

#### 1. Set up environment
Ensure all dependencies are installed (`pip install -r requirements.txt`).

#### 2. Running the script
Execute the script from the command line with the following arguments:
- `-d` or `--data_path`: Path to ECG dataset (required).
- `-o` or `--output_dir`: Directory to save outputs (required).
- `-w` or `--window_size`: Window size, default 64.
- `-l` or `--latent_dim`: Latent dimension size, default 16.
- `-e` or `--epochs`: Training epochs, default 10.
- `-b` or `--batch_size`: Batch size, default 64.
- `--device`: Training device (CPU/GPU), defaults to GPU if available.

If you're using the ECGData.mat example, no preprocessing is necessary.

#### Example command (using Metal acceleration on Mac)
```bash
python train.py -d data/ECGData.mat -o results --device mps -e 10
```


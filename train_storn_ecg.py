import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import DataLoader, TensorDataset

import configparser
from model.storn import build_STORN  # Import the STORN builder

##
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8  # Adjust these values to your preference
rcParams['figure.dpi'] = 100

tqdm.pandas()

##

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)


##

class STORNTrainer:
    def __init__(self, config, model, train_dataset, val_dataset, device='cpu'):
        self.config = config
        self.model = model
        self.device = device
        

def load_config(config_path="./config/cfg_storn.ini"):
    """Loads the configuration file."""
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def get_device(config):
    """Sets the device for computation."""
    return 'cuda' if torch.cuda.is_available() and config.getboolean('Training', 'use_cuda') else 'cpu'

def load_data():
    """Loads and splits the RAF dataset from a .mat file."""
    from dataset.utils.mat_to_npy_animator import MATConverterAnimator
    
    current_directory = os.path.dirname(os.path.abspath(__file__))
    mat_file_path = os.path.abspath(os.path.join(current_directory, '..', 'RAF_data', 'RAF_data.mat'))

    mat_data_reader = MATConverterAnimator()
    RAF_data = mat_data_reader.load_and_convert(mat_file_path)

    # Transpose and split dataset
    RAF_data = RAF_data.transpose(2, 0, 1)  # (1690, 61, 121)
    train_data, temp_data = train_test_split(RAF_data, test_size=0.3, shuffle=False)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=False)

    # Transpose back to original shape
    train_data = train_data.transpose(1, 2, 0)
    val_data = val_data.transpose(1, 2, 0)
    test_data = test_data.transpose(1, 2, 0)

    return train_data, val_data, test_data

def scale_dataset(data, scaler=None):
    """Scales a single dataset using MinMaxScaler."""
    # Transpose to (steps, features, time) for scaling
    data = data.transpose(2, 0, 1)  # Shape: (steps, 61, 121)

    # Reshape to 2D: (steps * 61, 121)
    data_2d = data.reshape(-1, data.shape[-1])

    # Fit or transform the data
    if scaler is None:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data_2d)
    else:
        scaled_data = scaler.transform(data_2d)

    # Reshape back to 3D and transpose to original shape
    scaled_data = scaled_data.reshape(data.shape).transpose(1, 2, 0)

    return scaled_data, scaler

def scaled_data_loader():
    """Loads and scales the RAF dataset."""
    train_data, val_data, test_data = load_data()

    # Scale train data and fit scaler
    train_data_scaled, scaler = scale_dataset(train_data)

    # Use the same scaler to transform validation and test data
    val_data_scaled, _ = scale_dataset(val_data, scaler)
    test_data_scaled, _ = scale_dataset(test_data, scaler)

    return train_data_scaled, val_data_scaled, test_data_scaled, scaler

def create_flatten_sequences(input_data, sequence_length, device):
    """
    Function to flatten 3D data and generate encoder, decoder, and target sequences on the specified device.

    Parameters:
    - input_data: NumPy array of shape (61, 121, 1690)
    - sequence_length: Length of each sequence (tau)
    - device: PyTorch device (e.g., 'cuda' or 'cpu')

    Returns:
    - encoder_input: Tensor of shape (num_samples, sequence_length, flattened_dim) on the specified device
    - decoder_input: Tensor of shape (num_samples, sequence_length, flattened_dim) on the specified device
    - target_output: Tensor of shape (num_samples, sequence_length, flattened_dim) on the specified device
    """

    # Step 1: Transpose to bring time to the first axis (optional)
    input_data = input_data.transpose(2, 0, 1)  # Now shape: (1690, 61, 121)

    # Step 2: Flatten spatial dimensions (61, 121) into a single dimension
    flattened_data = input_data.reshape(input_data.shape[0], -1)  # Shape: (1690, 7381)

    # Convert to PyTorch tensor and move to the specified device
    flattened_data = torch.tensor(flattened_data, dtype=torch.float32).to(device)

    # Calculate the number of samples
    num_samples = flattened_data.shape[0] - 2 * sequence_length - 1

    # Pre-allocate tensors on the specified device
    flattened_dim = flattened_data.shape[1]  # 7381 in your case
    encoder_input = torch.empty((num_samples, sequence_length, flattened_dim), device=device)
    decoder_input = torch.empty((num_samples, sequence_length, flattened_dim), device=device)
    target_output = torch.empty((num_samples, sequence_length, flattened_dim), device=device)

    # Generate and store sequences directly on the specified device
    for i in tqdm(range(num_samples)):
        encoder_input[i] = flattened_data[i:i + sequence_length, :]
        decoder_input[i] = flattened_data[i + sequence_length:i + 2 * sequence_length, :]
        target_output[i] = flattened_data[i + sequence_length + 1:i + 2 * sequence_length + 1, :]

    return encoder_input, decoder_input, target_output


if __name__ == "__main__":
    import sys
    
    from model.storn import build_STORN  # Import the STORN model builder

    # Step 1: Load config
    config = load_config()

    # Step 2: Set device
    device = get_device(config)

    # Step 3: Load scaled RAF data
    train_data_scaled, val_data_scaled, test_data_scaled, scaler = scaled_data_loader()

    sequence_length = config.getint('DataFrame', 'sequence_len')
    
    # Step 4: Create the sequential data to feed the model
    train_encoder_input, train_decoder_input, train_target_output = create_flatten_sequences(
        train_data_scaled, sequence_length, device
    )
    
    val_encoder_input, val_decoder_input, val_target_output = create_flatten_sequences(
        val_data_scaled, sequence_length, device
    )
    
    test_encoder_input, test_decoder_input, test_target_output = create_flatten_sequences(
        test_data_scaled, sequence_length, device
    )

    # Create TensorDatasets for training and testing
    train_dataset = TensorDataset(train_encoder_input, train_decoder_input, train_target_output)
    val_dataset = TensorDataset(val_encoder_input, val_decoder_input, val_target_output)
    test_dataset = TensorDataset(test_encoder_input, test_decoder_input, test_target_output)

    # Define batch size
    batch_size = config.getint('DataFrame', 'batch_size')

    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader =  DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Inspect the shape of the data inside the DataLoader
    for batch in train_loader:
        encoder_input, decoder_input, target_output = batch  # Unpack batch

        # Print shapes for verification
        print(f"Encoder Input Shape: {encoder_input.shape}")
        print(f"Decoder Input Shape: {decoder_input.shape}")
        print(f"Target Output Shape: {target_output.shape}")

        # Break after the first batch to avoid printing too much
        break


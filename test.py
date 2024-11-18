import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

from dataset.utils.mat_to_npy_animator import MATConverterAnimator

current_directory = os.path.dirname(os.path.abspath(__file__))
mat_file_path = os.path.abspath(os.path.join(current_directory, '..', 'RAF_data', 'RAF_data.mat'))

mat_data_reader = MATConverterAnimator()
RAF_data = mat_data_reader.load_and_convert(mat_file_path)

# Step 1: Transpose to bring time to the first axis (optional for consistency)
RAF_data = RAF_data.transpose(2, 0, 1)  # Now shape: (1690, 61, 121)

# Step 2: Flatten spatial dimensions (61, 121) into a single dimension
flattened_data = RAF_data.reshape(1690, -1)  # Shape: (1690, 7381)

# Assume flattened_data is available as a NumPy array with shape (1690, 7381)
flattened_data = torch.tensor(flattened_data, dtype=torch.float32)  # Convert to Torch tensor

# Move the data to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flattened_data = flattened_data.to(device)

# Sequence length
sequence_length = 100

# Calculate the number of samples
num_samples = len(flattened_data) - 2 * sequence_length - 1

# Pre-allocate tensors on GPU for encoder, decoder, and target inputs
encoder_input = torch.empty((num_samples, sequence_length, flattened_data.shape[1]), device=device)
decoder_input = torch.empty((num_samples, sequence_length, flattened_data.shape[1]), device=device)
target_output = torch.empty((num_samples, sequence_length, flattened_data.shape[1]), device=device)

# Generate and store sequences directly on GPU
for i in tqdm(range(num_samples)):
    encoder_input[i] = flattened_data[i:i + sequence_length, :]
    decoder_input[i] = flattened_data[i + sequence_length:i + 2 * sequence_length, :]
    target_output[i] = flattened_data[i + sequence_length + 1:i + 2 * sequence_length + 1, :]

print(f"Encoder Input Shape: {encoder_input.shape}")
print(f"Decoder Input Shape: {decoder_input.shape}")
print(f"Target Output Shape: {target_output.shape}")

print(f"Encoder Input Shape: {encoder_input.shape}")
print(f"Decoder Input Shape: {decoder_input.shape}")
print(f"Target Output Shape: {target_output.shape}")

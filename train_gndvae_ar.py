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
import torch.optim as optim

import configparser
from model.storn import build_STORN  # Import the STORN builder

# from data_preparation.utils.mat_to_npy_animator import MATConverterAnimator
from data_preparation.utils.ar_model import ARModel

import importlib
from model import gndvae
importlib.reload(gndvae)
GNDVAE = gndvae.GNDVAE
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
np.random.seed(RANDOM_SEED)


def load_config(config_path="./config/cfg_storn.ini"):
    """Loads the configuration file."""
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def get_device(config):
    """Sets the device for computation."""
    return 'cuda' if torch.cuda.is_available() and config.getboolean('Training', 'use_cuda') else 'cpu'

def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model

class STORNTrainer:
    def __init__(self, data_name, model, config, device='cpu'):
        self.data_name = data_name
        self.model = model
        self.config = config
        self.device = device
        self.batch_size = self.config.getint('DataFrame', 'batch_size')
        self.sequence_length = self.config.getint('DataFrame', 'sequence_len')
        self.beta = config.getfloat('Training', 'beta')
        

    def load_AR_data(self, rhoa=0.9):
        """Loads and splits the RAF dataset from a .mat file."""   
        num_variates = self.x_2d *  self.y_2d
        
        # Spectral norm must be less than 1
        self.ARA = ARModel.generate_random_AR(num_variates, self.ar_order, rhoa)
        # Set the threshold
        threshold = 3.3*np.sqrt(abs(np.mean(self.ARA)))
        # Set values below the threshold to zero
        self.ARA[abs(self.ARA) < threshold] = 0
        sns.heatmap(np.sum(np.abs(self.ARA), axis=2), cmap="Blues")
        # Add title and labels
        plt.title("Ground Truth Granger Causality")
        plt.ylabel("Target Sensor")
        plt.xlabel("Source Sensor")
        plt.show
        V = ARModel.generate_random_covariance(num_variates, float('inf'))
        ar_model = ARModel(self.ARA, V, self.num_samples)
        y, e = ar_model.generate_samples()
        data = y.reshape( self.y_2d, self.x_2d, self.num_samples)
        
        animation = False
        if animation == True:
            from matplotlib.animation import FuncAnimation
            """Generate and save an animation from the generated data."""        
            # Create figure and axis
            fig, ax = plt.subplots()
            heatmap = ax.imshow(data[:, :, 0], cmap='cividis', interpolation='nearest')
            cbar = fig.colorbar(heatmap)
            cbar.set_label('Value')

            def update(frame):
                """Update function for animation."""
                heatmap.set_array(data[:, :, frame])
                ax.set_title(f'Time: {frame}')
                return heatmap,

            # Create and save the animation
            interval=50
            ani = FuncAnimation(fig, update, frames=range(data.shape[-1]),
                                interval=interval, blit=True)
            
            # Construct the full path to the npy file relative to the script's directory
            npy_file_path = "/mnt/c/main_project/C_DVAE/dataset/generated_dataset/3d_data_heatmap.npy"
            output_path = os.path.splitext(npy_file_path)[0] + '.gif'
            ani.save(output_path, writer='imagemagick')
            print(f"Animation saved at: {output_path}")
            
        # Transpose and split dataset
        data = data.transpose(2, 0, 1)  
        self.train_data, temp_data = train_test_split(data, # The code `test_size` is defining a
        # variable named `test_size` in Python.
        test_size=0.3, shuffle=False)
        self.val_data, self.test_data = train_test_split(temp_data, test_size=0.5, shuffle=False)

        # Transpose back to original shape
        self.train_data = self.train_data.transpose(1, 2, 0)
        self.val_data = self.val_data.transpose(1, 2, 0)
        self.test_data = self.test_data.transpose(1, 2, 0)
        
        print("AR model generated and split into train, validation, and test sets.")
        print(f"  train_data: {self.train_data.shape}")
        print(f"  val_data: {self.val_data.shape}")
        print(f"  test_data: {self.test_data.shape}")
        print(f"---------------------------------------------------------------------")

        # return self.train_data, self.val_data, self.test_data

    # def load_RAF_data(self):
    #     """Loads and splits the RAF dataset from a .mat file."""
    #     from dataset.utils.mat_to_npy_animator import MATConverterAnimator
        
    #     current_directory = os.path.dirname(os.path.abspath(__file__))
    #     mat_file_path = os.path.abspath(os.path.join(current_directory, '..', 'RAF_data', 'RAF_data.mat'))

    #     mat_data_reader = MATConverterAnimator()
    #     RAF_data = mat_data_reader.load_and_convert(mat_file_path)

    #     # Transpose and split dataset
    #     RAF_data = RAF_data.transpose(2, 0, 1)  # (1690, 61, 121)
    #     self.train_data, temp_data = train_test_split(RAF_data, test_size=0.3, shuffle=False)
    #     self.val_data, self.test_data = train_test_split(temp_data, test_size=0.5, shuffle=False)

    #     # Transpose back to original shape
    #     self.train_data = self.train_data.transpose(1, 2, 0)
    #     self.val_data = self.val_data.transpose(1, 2, 0)
    #     self.test_data = self.test_data.transpose(1, 2, 0)

        # return self.train_data, self.val_data, self.test_data
    
    @staticmethod
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

    def scaled_data_loader(self):
        """Loads and scales the RAF dataset."""
        if self.data_name == 'AR':
            self.x_2d = 4
            self.y_2d = 4
            self.num_samples = 10000
            self.ar_order = 10
            self.load_AR_data()
        elif self.data_name == 'RAF':
            self.load_RAF_data()
        else:
            assert False, "data type is not valid!"

        # Scale train data and fit scaler
        self.train_data_scaled, self.scaler = self.scale_dataset(self.train_data)

        # Use the same scaler to transform validation and test data
        self.val_data_scaled, _ = self.scale_dataset(self.val_data, self.scaler)
        self.test_data_scaled, _ = self.scale_dataset(self.test_data, self.scaler)
        
        print("Data scaled to range (-1, 1).")
        print(f"  train_data_scaled: {self.train_data_scaled.shape}")
        print(f"  val_data_scaled: {self.val_data_scaled.shape}")
        print(f"  test_data_scaled: {self.test_data_scaled.shape}")
        print(f"---------------------------------------------------------------------")

        # return train_data_scaled, val_data_scaled, test_data_scaled, scaler
    
    def create_flatten_sequences(self, input_data):
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
        flattened_data = torch.tensor(flattened_data, dtype=torch.float32).to(self.device)

        # Calculate the number of samples
        num_samp = flattened_data.shape[0] - 2 * self.sequence_length - 1

        # Pre-allocate tensors on the specified device
        flattened_dim = flattened_data.shape[1] 
        encoder_input = torch.empty((num_samp, self.sequence_length, flattened_dim), device=self.device)
        decoder_input = torch.empty((num_samp, self.sequence_length, flattened_dim), device=self.device)
        target_output = torch.empty((num_samp, self.sequence_length, flattened_dim), device=self.device)

        # Generate and store sequences directly on the specified device
        for i in tqdm(range(num_samp)):
            encoder_input[i] = flattened_data[i:i + self.sequence_length, :]
            decoder_input[i] = flattened_data[i + self.sequence_length:i + 2 * self.sequence_length, :]
            target_output[i] = flattened_data[i + self.sequence_length + 1:i + 2 * self.sequence_length + 1, :]
        
        
        return encoder_input, decoder_input, target_output

    def load_and_prepare_data(self, info_disp=False):
        self.scaled_data_loader()
        
        # Step 4: Create the sequential data to feed the model
        train_encoder_input, train_decoder_input, train_target_output = self.create_flatten_sequences(self.train_data_scaled)
        
        if info_disp:
            print(f"Flattened data sequences generated with sequence length = {self.sequence_length}.")
            print(f" Train:")
            print(f"  train_encoder_input: {train_encoder_input.shape}")
            print(f"  train_decoder_input: {train_decoder_input.shape}")
            print(f"  train_target_output: {train_target_output.shape}")
            print(f"----------------------------------------")

        val_encoder_input, val_decoder_input, val_target_output = self.create_flatten_sequences(self.val_data_scaled)
        if info_disp:
            print(f" Validation:")
            print(f"  val_encoder_input: {val_encoder_input.shape}")
            print(f"  val_decoder_input: {val_decoder_input.shape}")
            print(f"  val_target_output: {val_target_output.shape}")
            print(f"----------------------------------------")
        
        test_encoder_input, test_decoder_input, test_target_output = self.create_flatten_sequences(self.test_data_scaled)
        if info_disp:
            print(f" Test:")
            print(f"  test_encoder_input: {test_encoder_input.shape}")
            print(f"  test_decoder_input: {test_decoder_input.shape}")
            print(f"  test_target_output: {test_target_output.shape}")
            print(f"---------------------------------------------------------------------")
        
        # Create TensorDatasets for training and testing
        self.train_dataset = TensorDataset(train_encoder_input, train_decoder_input, train_target_output)
        self.val_dataset = TensorDataset(val_encoder_input, val_decoder_input, val_target_output)
        self.test_dataset = TensorDataset(test_encoder_input, test_decoder_input, test_target_output)

        # Define batch size
        self.batch_size = self.config.getint('DataFrame', 'batch_size')

        # Create DataLoaders for training and testing
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)
        self.val_loader =  DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        
        if info_disp:
            print(f"Data loader generated with batch size = {self.batch_size}.")
            for batch_idx, (encoder_input, decoder_input, target_output) in enumerate(self.train_loader):
                print(f" Train:")
                print(f"  Encoder input size = {encoder_input.size()}")
                print(f"  Decoder input size = {decoder_input.size()}")
                print(f"  Target output size = {target_output.size()}")
                print(f"----------------------------------------")
                break  

            for batch_idx, (encoder_input, decoder_input, target_output) in enumerate(self.val_loader):
                print(f" Validation:")
                print(f"  Encoder input size = {encoder_input.size()}")
                print(f"  Decoder input size = {decoder_input.size()}")
                print(f"  Target output size = {target_output.size()}")
                print(f"----------------------------------------")
                break  

            for batch_idx, (encoder_input, decoder_input, target_output) in enumerate(self.test_loader):
                print(f" Test:")
                print(f"  Encoder input size = {encoder_input.size()}")
                print(f"  Decoder input size = {decoder_input.size()}")
                print(f"  Target output size = {target_output.size()}")
                print(f"---------------------------------------------------------------------")
                break  

    def train(self):
        # Load training configurations
        lr = self.config.getfloat('Training', 'lr')
        epochs = self.config.getint('Training', 'epochs')
        early_stop_patience = self.config.getint('Training', 'early_stop_patience')

        # Initialize optimizer
        optimizer = optim.Adam(unwrap_model(self.model).parameters(), lr=lr)
        
        # Track validation loss for early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0.0

            for batch_idx, (encoder_input, decoder_input, target_output) in enumerate(self.train_loader):
                # Move inputs and target to the device
                encoder_input = encoder_input.to(self.device)
                decoder_input = decoder_input.to(self.device)
                target_output = target_output.to(self.device)

                # Forward pass with both encoder and decoder inputs
                reconstructed, z_mean, z_logvar, _ = self.model(encoder_input, decoder_input)

                # Calculate loss
                loss_tot, loss_recon, loss_KLD = unwrap_model(self.model).get_loss(
                    target_output, reconstructed, z_mean, z_logvar, 
                    seq_len=encoder_input.size(1), 
                    batch_size=encoder_input.size(0), 
                    beta=self.beta
                )

                # Backward pass and optimize
                optimizer.zero_grad()
                loss_tot.backward()
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(unwrap_model(self.model).parameters(), max_norm=1.0)

                optimizer.step()

                # Update total loss
                total_train_loss += loss_tot.item()

            # Calculate average training loss
            avg_train_loss = total_train_loss / len(self.train_loader)
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Recon Loss: {loss_recon.item():.4f}, KL Loss: {loss_KLD.item():.4f}')

            # Validation step
            val_loss = self.validate()
            print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}')

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Define the current directory and save path
                current_directory = os.path.dirname(os.path.abspath(__file__))
                saved_path = os.path.join(current_directory, "saved_models", "best_model_nov_19.pth")
                # Save the model's state_dict to the specified path
                torch.save(unwrap_model(self.model).state_dict(), saved_path)
                print("Saved best model.")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print("Early stopping triggered.")
                    break

    def validate(self):
        self.model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for encoder_input, decoder_input, target_output in self.val_loader:
                encoder_input = encoder_input.to(self.device)
                decoder_input = decoder_input.to(self.device)
                target_output = target_output.to(self.device)

                # Forward pass with both encoder and decoder inputs
                reconstructed, z_mean, z_logvar, _ = self.model(encoder_input, decoder_input)

                # Calculate loss
                loss_tot, _, _ = unwrap_model(self.model).get_loss(
                    target_output, reconstructed, z_mean, z_logvar, 
                    seq_len=encoder_input.size(1), 
                    batch_size=encoder_input.size(0), 
                    beta=self.beta
                )
                total_val_loss += loss_tot.item()

        return total_val_loss / len(self.val_loader)

    import torch
    import os

    def load_checkpoint(self):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(current_directory, "saved_models", "best_model_nov_19.pth")
        
        if os.path.exists(checkpoint_path):
            # Use unwrap_model to handle DataParallel
            unwrap_model(self.model).load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print("Checkpoint loaded successfully.")
        else:
            print("No checkpoint found. Please check the saved path or run training first.")


    def test(self):
        self.model.eval()
        total_test_loss = 0.0
        all_reconstructed = []
        all_attention_weights = []

        with torch.no_grad():
            for encoder_input, decoder_input, target_output in self.test_loader:
                encoder_input = encoder_input.to(self.device)
                decoder_input = decoder_input.to(self.device)
                target_output = target_output.to(self.device)

                # Forward pass with encoder and decoder inputs
                reconstructed, z_mean, z_logvar, attention_weights = self.model(encoder_input, decoder_input)

                # Calculate loss
                loss_tot, loss_recon, loss_KLD = unwrap_model(self.model).get_loss(
                    target_output, reconstructed, z_mean, z_logvar, 
                    seq_len=encoder_input.size(1), 
                    batch_size=encoder_input.size(0), 
                    beta=self.beta
                )
                total_test_loss += loss_tot.item()

                # Store results for analysis
                all_reconstructed.append(reconstructed.cpu())
                all_attention_weights.append(attention_weights.cpu())

        avg_test_loss = total_test_loss / len(self.test_loader)
        print(f"Test Loss: {avg_test_loss:.4f}, Recon Loss: {loss_recon.item():.4f}, KL Loss: {loss_KLD.item():.4f}")

        return all_reconstructed, all_attention_weights

# Running the test function
if __name__ == "__main__":
    # Load the class to 
    config = load_config('./config/cfg_gndvae.ini')
    device = get_device(config)
    print(config.getint('Network', 'x_dim'))
    # Initialize the model
    model = GNDVAE(
        x_dim=config.getint('Network', 'x_dim'),
        z_dim=config.getint('Network', 'z_dim'),
        encoder_config={
            'x_dim': config.getint('Network', 'x_dim'),
            'dense_input_h': [512, 128],
            'dim_RNN_h': config.getint('Network', 'dim_RNN_h'),
            'num_RNN_h': config.getint('Network', 'num_RNN_h'),
            'dense_hx_g': [256, 128],
            'dim_RNN_g': config.getint('Network', 'dim_RNN_g'),
            'num_RNN_g': config.getint('Network', 'num_RNN_g'),
            'z_dim': config.getint('Network', 'z_dim'),
            'activation': config.get('Network', 'activation'),
            'dropout_p': config.getfloat('Network', 'dropout_p'),
            'device': device
        },
        decoder_config={
            'x_dim': config.getint('Network', 'x_dim'),
            'z_dim': config.getint('Network', 'z_dim'),
            'dense_z_s': [64, 32],
            'tau': config.getint('DataFrame', 'sequence_len'),
            'activation': config.get('Network', 'activation'),
            'dropout_p': config.getfloat('Network', 'dropout_p'),
            'device': device
        },
        activation=config.get('Network', 'activation'),
        dropout_p=config.getfloat('Network', 'dropout_p'),
        device=device
    )
    
    # Enable multi-GPU training if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)

    # Move the model to the selected device
    model = model.to(device)
    
    tau = config.getint('DataFrame', 'sequence_len')
    x_dim = config.getint('Network', 'x_dim')
    z_dim = config.getint('Network', 'z_dim')
    dense_z_s = [64, 32]
    data_name = 'AR'
    trainer_class = STORNTrainer(data_name, model, config, device)
    
    # Load and preprocess data
    trainer_class.load_and_prepare_data()
    
    Training_phase = True
    if Training_phase:
        # Train the model
        trainer_class.train()
    else:
        # Load the model
        trainer_class.load_checkpoint()
        
    # trained_influence_matrices = torch.zeros(tau, x_dim, x_dim + z_dim, dense_z_s[0])
    # for i in range(x_dim):  # Loop over target features
    #     for k in range(tau):  # Loop over lags
    #         trained_influence_matrices[k, i, :, :] = model.decoder.intermediate_layers[i][k].weight.T
    
    # influence_matrix = torch.norm(trained_influence_matrices, p=2, dim=3)  # Shape: [100, 16, 48]
    
    # influence_matrix_sum = torch.sum(influence_matrix[0:10], dim=0)  # Shape: [16, 48]
    # influence_matrix_max = torch.max(influence_matrix, dim=0).values  # Take the max over lags
    # threshold = 0  # Adjust based on analysis
    # influence_matrix_thresholded = influence_matrix_sum.clone()
    # influence_matrix_thresholded[influence_matrix_sum < threshold] = 0
    
    # input_feature_influence = influence_matrix_thresholded[:, :16].T  # Shape: [16, 16]
    # plt.show()
    # sns.heatmap(input_feature_influence.detach().cpu().numpy(), cmap="Blues", cbar=True)
    # plt.xlabel("Input Features (Sources)")
    # plt.ylabel("Target Features")
    # plt.title("Influence of Input Features on Targets")
    # plt.show()
from torch import nn
import torch
from collections import OrderedDict

class Encoder(nn.Module):
    def __init__(self, 
                x_dim,               # Input dimension (flattened data size)
                dense_input_h,       # List of MLP layer sizes for input preprocessing
                dim_RNN_h,           # Hidden size of first RNN (h_t)
                num_RNN_h,           # Number of layers in first RNN
                dense_hx_g,          # List of MLP sizes between RNNs (h_t + x_t -> g_t)
                dim_RNN_g,           # Hidden size of second RNN (g_t)
                num_RNN_g,           # Number of layers in second RNN
                z_dim=16,            # Latent dimension size
                activation='tanh',   # Activation function type
                dropout_p=0.1,       # Dropout probability
                device='cpu'):       # Device: 'cpu' or 'cuda'
        super().__init__()

        # Store parameters
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.dim_RNN_h = dim_RNN_h
        self.num_RNN_h = num_RNN_h
        self.dim_RNN_g = dim_RNN_g
        self.num_RNN_g = num_RNN_g
        self.dropout_p = dropout_p
        self.device = device

        # Activation function selection
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Invalid activation type!")

        # Store MLP layer dimensions
        self.dense_input_h = dense_input_h
        self.dense_hx_g = dense_hx_g

        # Build the encoder model components
        self.build()

    def build(self):
        ###############################
        #### Input MLP Layer (h_t) ####
        ###############################
        # Add MLP layer for input to forward RNN
        dic_layers = OrderedDict()

        if len(self.dense_input_h) > 0:
            for n in range(len(self.dense_input_h)):
                if n == 0:
                    dic_layers[f'linear{n}'] = nn.Linear(self.x_dim, self.dense_input_h[n])
                else:
                    dic_layers[f'linear{n}'] = nn.Linear(self.dense_input_h[n - 1], self.dense_input_h[n])
                dic_layers[f'activation{n}'] = self.activation
                dic_layers[f'dropout{n}'] = nn.Dropout(p=self.dropout_p)

            dim_input_h = self.dense_input_h[-1]  # Output size of last MLP layer
        else:
            dic_layers['Identity'] = nn.Identity()
            dim_input_h = self.x_dim  # No change if no MLP

        self.mlp_input_h = nn.Sequential(dic_layers)

        #######################
        #### First RNN (h_t) ####
        #######################
        # Forward RNN processing input h_t
        # self.rnn_h = nn.LSTM(dim_input_h, self.dim_RNN_h, self.num_RNN_h, batch_first=True)
        self.rnn_h = nn.GRU(dim_input_h, self.dim_RNN_h, self.num_RNN_h, batch_first=True)

        ######################
        #### Inference MLP ####
        ######################
        # Prepare concatenated h_t + x_t for the backward RNN (g_t)
        dic_layers = OrderedDict()
        dim_hx_g = self.dim_RNN_h + self.x_dim  # Concatenated size

        if len(self.dense_hx_g) > 0:
            for n in range(len(self.dense_hx_g)):
                if n == 0:
                    dic_layers[f'linear{n}'] = nn.Linear(dim_hx_g, self.dense_hx_g[n])
                else:
                    dic_layers[f'linear{n}'] = nn.Linear(self.dense_hx_g[n - 1], self.dense_hx_g[n])
                dic_layers[f'activation{n}'] = self.activation
                dic_layers[f'dropout{n}'] = nn.Dropout(p=self.dropout_p)

            dim_hx_g_out = self.dense_hx_g[-1]
        else:
            dic_layers['Identity'] = nn.Identity()
            dim_hx_g_out = dim_hx_g

        self.mlp_hx_g = nn.Sequential(dic_layers)

        #########################
        #### Second RNN (g_t) ####
        #########################
        # Backward RNN processing (g_t)
        # self.rnn_g = nn.LSTM(dim_hx_g_out, self.dim_RNN_g, self.num_RNN_g, batch_first=True)
        self.rnn_g = nn.GRU(dim_hx_g_out, self.dim_RNN_g, self.num_RNN_g, batch_first=True)
        
        ##########################
        #### Latent Space (z) ####
        ##########################
        # MLP for Mean (mu) and Log-Variance (logvar)
        self.mlp_mean = nn.Linear(self.dim_RNN_g, self.z_dim)   # Output mean (mu)
        self.mlp_logvar = nn.Linear(self.dim_RNN_g, self.z_dim)  # Output log-variance (logvar)
        
        
    def reparameterize(self, mean, logvar):
        """Reparameterization trick to sample from N(mean, var) using standard Gaussian."""
        std = torch.exp(0.5 * logvar)  # Compute standard deviation
        eps = torch.randn_like(std)    # Random noise with the same shape as std
        return mean + eps * std        # Sampled z = mean + eps * std

    def forward(self, x):
        # Step 1: Run through the forward and backward RNNs (same as before)
        g_seq = self.run_through_rnns(x)  # g_t: [batch_size, sequence_length, dim_RNN_g]

        # Step 2: Compute mean and log-variance for each time step
        mean = self.mlp_mean(g_seq)     # Shape: [batch_size, sequence_length, z_dim]
        logvar = self.mlp_logvar(g_seq)  # Shape: [batch_size, sequence_length, z_dim]

        # Step 3: Reparameterize to get latent variable z
        z = self.reparameterize(mean, logvar)  # Shape: [batch_size, sequence_length, z_dim]

        return z, mean, logvar  # Return z, mean, and logvar
    
    def run_through_rnns(self, x):
        """Helper function to run through the forward and backward RNNs."""
        # Step 1: Preprocess input with MLP layer
        x_processed = self.mlp_input_h(x)  # Shape: [batch_size, sequence_length, dim_RNN_h]

        # Step 2: Forward RNN (h_t)
        h_seq, _ = self.rnn_h(x_processed)  # Shape: [batch_size, sequence_length, dim_RNN_h]

        # Step 3: Concatenate h_t with original input x_t
        hx_concat = torch.cat((h_seq, x), dim=-1)  # Shape: [batch_size, sequence_length, dim_RNN_h + x_dim]

        # Step 4: Process concatenated input with MLP
        hx_processed = self.mlp_hx_g(hx_concat)  # Shape: [batch_size, sequence_length, dim_hx_g_out]

        # Step 5: Backward RNN (g_t)
        g_seq, _ = self.rnn_g(torch.flip(hx_processed, [1]))  # Reverse along sequence length
        g_seq = torch.flip(g_seq, [1])  # Flip back to original order

        return g_seq  # Output of encoder: g_t

class MultiHeadAttentionDecoder(nn.Module):
    def __init__(self, x_dim, z_dim, dense_z_s, tau, activation='tanh', dropout_p=0.1, device='cpu'):
        super().__init__()
        
        # Store parameters
        self.x_dim = x_dim         # Dimension of each input feature
        self.z_dim = z_dim         # Latent dimension
        self.dense_z_s = dense_z_s # Size of the intermediate layer output
        self.tau = tau             # Number of past time steps to consider
        self.device = device       # Device (e.g., 'cpu' or 'cuda')
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Invalid activation type!")
        
        self.dropout_p = dropout_p

        # Nested ModuleList with separate weights for each target feature and lag
        self.intermediate_layers = nn.ModuleList([
            nn.ModuleList([nn.Linear(self.x_dim + self.z_dim, self.dense_z_s[0]) for _ in range(tau)])
            for _ in range(x_dim)
        ])

        # Separate output MLP for each target feature and each lag
        self.output_mlp = nn.ModuleList([
            nn.ModuleList([nn.Linear(self.dense_z_s[0], 1) for _ in range(tau)])  # One output layer per lag
            for _ in range(x_dim)  # One set of output layers per target feature
        ])
        
    def forward(self, z, x_seq):
        batch_size, seq_len, _ = z.shape
        
        # Initialize output tensor and influence matrices (A)
        output = torch.zeros(batch_size, seq_len, self.x_dim).to(self.device)
        
        # Influence matrices A to store causal weights for each target, lag, and feature
        influence_matrices = torch.zeros(self.tau, self.x_dim, self.x_dim + self.z_dim, self.dense_z_s[0]).to(self.device)

        # Update influence matrices outside the time loop to avoid redundant updates
        for i in range(self.x_dim):  # Loop over target features
            for k in range(self.tau):  # Loop over lags
                influence_matrices[k, i, :, :] = self.intermediate_layers[i][k].weight.T

        # Loop over each time step to calculate predictions
        for t in range(seq_len):
            z_t = z[:, t, :]  # Current latent variable [batch_size, z_dim]

            # Store predictions for each target feature at time step t
            predictions = []
            
            for i in range(self.x_dim):  # Loop over target features
                prediction_i_lags = []
                
                for k in range(self.tau):
                    if t - k - 1 >= 0:  # Only proceed if there are valid past observations
                        x_past = x_seq[:, t - k - 1, :]  # Lagged input x_{t-k} [batch_size, x_dim]
                        combined_input = torch.cat((x_past, z_t), dim=-1)  # [batch_size, x_dim + z_dim]
                        
                        # Pass through intermediate layer for (target, lag) using i and k
                        inter_output = self.intermediate_layers[i][k](combined_input)  # [batch_size, dense_z_s[0]]
                        # Apply output layer specific to (target feature, lag)
                        prediction_i_k = self.output_mlp[i][k](inter_output).squeeze(-1)  # [batch_size]
                        prediction_i_lags.append(prediction_i_k)
                
                if prediction_i_lags:
                    prediction_i = torch.stack(prediction_i_lags, dim=1).sum(dim=1)  # [batch_size]
                    predictions.append(prediction_i)
                else:
                    predictions.append(torch.zeros(batch_size).to(self.device))  # Append zero if no valid lags
                        
            # Combine predictions for each target feature to form the final output at time step t
            output[:, t, :] = torch.stack(predictions, dim=-1)  # [batch_size, x_dim]
        
        return output, influence_matrices  # Return the final output and all influence matrices
    
    def l1_regularization(self):
        """Calculate L1 norm of the weights in intermediate layers and output MLPs."""
        l1_norm = 0.0
        for i in range(self.x_dim):
            for k in range(self.tau):
                l1_norm += torch.sum(torch.abs(self.intermediate_layers[i][k].weight))
                l1_norm += torch.sum(torch.abs(self.output_mlp[i][k].weight))
        return l1_norm

class GNDVAE(nn.Module):
    def __init__(self, 
                x_dim, z_dim,  # Input and latent dimensions
                encoder_config, decoder_config,  # Encoder and decoder configs
                activation='tanh', 
                dropout_p=0.1, 
                device='cpu'):
        super().__init__()

        # Store parameters
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device

        # Build the encoder and decoder
        self.encoder = Encoder(**encoder_config).to(device)
        self.decoder = MultiHeadAttentionDecoder(**decoder_config).to(device)

    def reparameterize(self, mean, logvar):
        """Reparameterization trick to sample from N(mean, var)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x_enc, x_dec):
        """
        Forward pass through the GDVA model.
        Args:
            x: Input tensor [batch_size, sequence_length, x_dim]
        Returns:
            Reconstructed output, latent variables, and attention weights.
        """
        # Step 1: Encoder - Get latent variables (z), mean, and logvar
        z, mean, logvar = self.encoder(x_enc)

        # Step 2: Reparameterization - Sample z from latent space
        z_sampled = self.reparameterize(mean, logvar)

        # Step 3: Decoder - Generate reconstructed sequence
        reconstructed, weights_per_timestep = self.decoder(z_sampled, x_dec)

        return reconstructed, mean, logvar, weights_per_timestep

    def get_loss(self, x, y, z_mean, z_logvar, seq_len, batch_size, beta=1, lambda_l1=0.01):
        # Ensure all tensors are on the same device
        x = x.to(self.device)
        y = y.to(self.device)
        z_mean = z_mean.to(self.device)
        z_logvar = z_logvar.to(self.device)

        # Reconstruction loss (you can replace with mse_loss or binary_cross_entropy if needed)
        loss_recon = torch.nn.functional.mse_loss(y, x, reduction='sum')

        # KL divergence loss
        loss_KLD = -0.5 * torch.sum(z_logvar - z_logvar.exp() - z_mean.pow(2) + 1)

        # Normalize the losses
        loss_recon = loss_recon / (batch_size * seq_len)
        loss_KLD = loss_KLD / (batch_size * seq_len)

        # Calculate L1 regularization for decoder weights
        l1_loss = self.decoder.l1_regularization()  # Ensure your decoder has this method

        # Total loss with beta weighting
        loss_tot = loss_recon + beta * loss_KLD + lambda_l1 * l1_loss

        # Return the losses as tensors (do not convert to .item() here)
        return loss_tot, loss_recon, loss_KLD

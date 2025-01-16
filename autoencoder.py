import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        # add positional encoding
        x = x + self.pe[:, :seq_len, :]
        return x

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_size=1440, init_layer_size=4, latent_dim=64, nheads=4, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=32, dropout=0.2):
        """
        Transformer-based autoencoder for 1D time-series data.

        Parameters:
            input_size (int): Length of the time series (e.g. 1440 for 1-day of 1-min intervals).
            init_layer_size (int): Number of features at each timestep.
            latent_dim (int): Dimension of the latent (compressed) representation. This also acts as d_model.
            nheads (int): Number of attention heads in the Transformer.
            num_encoder_layers (int): Number of transformer encoder layers.
            num_decoder_layers (int): Number of transformer decoder layers.
            dim_feedforward (int): Dimensionality of the feedforward networks within the transformer layers.
            dropout (float): Dropout probability in the transformer layers.
        """
        super().__init__()
        self.input_size = input_size
        self.init_layer_size = nheads = init_layer_size
        self.latent_dim = latent_dim
        self.d_model = latent_dim  # Using latent_dim as the model dimension

        self.attn_implementation="flash_attention_2"
        self.use_nested_tensor = True

        # Project input features to d_model
        self.input_projection = nn.Linear(init_layer_size, self.d_model)

        # Positional encoding for encoder and decoder
        self.pos_encoder = PositionalEncoding(d_model=self.d_model, max_len=input_size)
        self.pos_decoder = PositionalEncoding(d_model=self.d_model, max_len=input_size)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nheads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first= True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=nheads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first = True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Projection from d_model back to init_layer_size
        self.output_projection = nn.Linear(self.d_model, init_layer_size)

        # Learnable queries for the decoder to reconstruct the sequence
        self.query_embedding = nn.Parameter(torch.randn(input_size, self.d_model))

    def encode(self, x):
        """
        Encode the input sequence into a latent vector.
        
        Input:
            x: (batch_size, init_layer_size, input_size)
        
        Output:
            latent: (batch_size, latent_dim)
        """
        # Reshape to (batch_size, input_size, init_layer_size)
        x = x.permute(0, 2, 1)  # Now: (batch, seq_len, features)

        # Project input features to model dimension
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder expects shape (seq_len, batch, d_model)
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        encoded_seq = self.transformer_encoder(x)  # (seq_len, batch, d_model)

        # Convert back to (batch, seq_len, d_model)
        encoded_seq = encoded_seq.transpose(0, 1)

        # Pool the sequence to get a single latent vector
        # Simple average pooling over the time dimension
        latent = encoded_seq.mean(dim=1)  # (batch, d_model) -> (batch, latent_dim)
        return latent

    def decode(self, latent):
        """
        Decode the latent vector back into the original sequence.
        
        Input:
            latent: (batch_size, latent_dim)
        
        Output:
            reconstructed: (batch_size, init_layer_size, input_size)
        """
        batch_size = latent.size(0)

        # We will create a set of queries for the decoder, one per timestep
        queries = self.query_embedding.unsqueeze(1).repeat(1, batch_size, 1)  # (seq_len, batch, d_model)

        # Positional encoding for decoder queries
        queries = queries.transpose(0, 1)  # (batch, seq_len, d_model)
        queries = self.pos_decoder(queries)
        queries = queries.transpose(0, 1)  # (seq_len, batch, d_model)

        # Memory for decoder:
        # Since we only have a single latent vector, we replicate it across all timesteps as memory.
        memory = latent.unsqueeze(0).repeat(self.input_size, 1, 1)  # (seq_len, batch, d_model)

        # Decode
        decoded_seq = self.transformer_decoder(queries, memory)  # (seq_len, batch, d_model)

        # Project back to init_layer_size
        decoded_seq = self.output_projection(decoded_seq)  # (seq_len, batch, init_layer_size)

        # Rearrange to (batch, init_layer_size, input_size)
        decoded_seq = decoded_seq.permute(1, 2, 0)  # (batch, init_layer_size, input_size)
        return decoded_seq

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed

class UnetAutoencoder(nn.Module):
    def __init__(self, input_size, latent_dim, init_layer_size = 1):
        """
        Initialize the Conv1D Autoencoder.

        Args:
            input_size (int): Size of the input sequence (timesteps).
            latent_dim (int): Dimensionality of the latent space.
        """
        super(UnetAutoencoder, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        # Encoder
        # (B, init_layer_size, T_in)
        self.enc1 = nn.Sequential(
            nn.Conv1d(init_layer_size, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True)
        )
        # (B, 64, T_in/2)

        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True)
        )
        # (B, 128, T_in/4)

        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True)
        )
        # (B, 256, T_in/8)

        self.enc4 = nn.Sequential(
            nn.Conv1d(256, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(inplace=True)
        )
        # (B, latent_dim, T_in/16) - latent representation

        # Decoder (mirroring encoder)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True)
        )
        # After dec4: (B, 256, T_in/8)

        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(256+256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True)
        )
        # After dec3: (B, 128, T_in/4)

        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(128+128, 64, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True)
        )
        # After dec2: (B, 64, T_in/2)

        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(64+64, init_layer_size, kernel_size=9, stride=2, padding=4, output_padding=1)
        )
        # After dec1: (B, init_layer_size, T_in) ~ reconstructed output

    def forward(self, x):
        # Encoder
        encoded1, encoded2, encoded3, encoded4 = self.encode(x)
        # Decoder with skip connections
        decoded = self.decode(encoded1, encoded2, encoded3, encoded4)
        return decoded

    def encode(self, data):
        """Encode input data into the latent space."""
        e1 = self.enc1(data)    # (B, 64, T/2)
        e2 = self.enc2(e1)   # (B, 128, T/4)
        e3 = self.enc3(e2)   # (B, 256, T/8)
        e4 = self.enc4(e3)   # (B, latent_dim, T/16)
        return e1,e2,e3,e4

    def decode(self, e1, e2, e3, e4):
        """Decode latent representation back to the original space."""
        
        d4 = self.dec4(e4)             # (B, 256, T/8)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))  # (B, 128, T/4)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # (B, 64, T/2)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # (B, init_layer_size, T)
        return d1

class DenseAutoencoder(nn.Module):
    def __init__(self, input_size=1440, latent_dim=64, init_layer_size=4):
        """
        A fully-connected (dense) autoencoder that compresses time-series data.

        Parameters:
            input_size (int): Number of timesteps in the input sequence (e.g. 1440 for a day at 1-min intervals).
            init_layer_size (int): Number of features at each timestep (e.g. price, volume, etc.).
            latent_dim (int): Dimensionality of the latent (compressed) representation.

        This model flattens the entire sequence of (features x timesteps) into a single vector, 
        passes it through a sequence of linear layers for encoding, and then reconstructs it.
        """
        super(DenseAutoencoder, self).__init__()
        self.input_size = input_size
        self.init_layer_size = init_layer_size
        
        # The total number of input values per sample. For example, if init_layer_size=4 and input_size=1440,
        # input_dim = 4 * 1440 = 5760.
        self.input_dim = input_size * init_layer_size

        # The encoder will map from input_dim (e.g., 5760) down to latent_dim (e.g. 64).
        # Intermediate layers help the model learn complex transformations.
        # Encoder with dropout regularization
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.LayerNorm(512),           # Reduce from 5760 to 1024
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.2),                  # Dropout with probability 0.5
            
            nn.Linear(512, 256),         # Reduce from 5760 to 1024
            nn.LayerNorm(256),  
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.2),           # Dropout with probability 0.5

            nn.Linear(256, 128),         # Reduce from 5760 to 1024
            nn.LayerNorm(128),  
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.2),                  # Dropout with probability 0.5
            
            nn.Linear(128, latent_dim)
        )

        # Decoder with dropout regularization
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),           # Reduce from 5760 to 1024
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.2),                  # Dropout with probability 0.5
            
            nn.Linear(128, 256),         # Reduce from 5760 to 1024
            nn.LayerNorm(256),  
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.2),           # Dropout with probability 0.5

            nn.Linear(256, 512),         # Reduce from 5760 to 1024
            nn.LayerNorm(512),  
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.2),                  # Dropout with probability 0.5
            
            nn.Linear(512, self.input_dim)
        )

    def forward(self, x):
        """
        Forward pass of the autoencoder.
        
        Expected input shape: (batch_size, init_layer_size, input_size)
        For example: (batch_size, 4, 1440)

        Steps:
        1. Flatten the data: (batch_size, 4, 1440) -> (batch_size, 5760)
        2. Encode through linear layers to latent space: (batch_size, 5760) -> (batch_size, latent_dim)
        3. Decode back to original dimensionality: (batch_size, latent_dim) -> (batch_size, 5760)
        4. Reshape to (batch_size, 4, 1440) to match original input shape.
        """

        batch_size = x.size(0)  # Get the number of samples in the batch

        # Flatten the input from (batch, features, timesteps) to (batch, features*timesteps)
        x_flat = x.view(batch_size, -1)  # Now (batch_size, input_dim)

        # Encode into the latent space (reducing dimensionality)
        latent = self.encoder(x_flat)  # (batch_size, latent_dim)

        # Decode from latent space back to full dimension
        reconstructed_flat = self.decoder(latent)  # (batch_size, input_dim)

        # Reshape back to (batch_size, init_layer_size, input_size)
        reconstructed = reconstructed_flat.view(batch_size, self.init_layer_size, self.input_size)

        return reconstructed

    def encode(self, x):
        """
        Encode the input data into the latent representation.

        Input shape: (batch_size, init_layer_size, input_size)
        Output shape: (batch_size, latent_dim)
        """
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  # Flatten
        return self.encoder(x_flat)

    def decode(self, latent):
        """
        Decode from the latent space back to the original space.

        Input shape: (batch_size, latent_dim)
        Output shape: (batch_size, init_layer_size, input_size)
        """
        batch_size = latent.size(0)
        reconstructed_flat = self.decoder(latent)   # (batch_size, input_dim)
        reconstructed = reconstructed_flat.view(batch_size, self.init_layer_size, self.input_size)
        return reconstructed
  

def load_csv_for_autoencoder(csv_file, feature_cols = ["mid_price","imbalance_take_price","imbalance_feed_price","imbalance_regulation_state","month","day_of_week","hour_of_day"], 
                             window_size = 60*24, stride = 30, batch_size = 64):
    """
    Creates a PyTorch DataLoader from a CSV file containing time series data.
    
    Args:
        csv_file (str): Path to the CSV file containing the data
        feature_cols (list): List of column names to use as features
        window_size (int): Size of the sliding window for sequence creation
        stride (int): Number of steps to move the window forward
        batch_size (int): Size of batches for the DataLoader
    
    Returns:
        DataLoader: PyTorch DataLoader containing the sequences
        
    Raises:
        ValueError: If the data has fewer samples than the window_size
    
    Notes:
        Time columns ('month', 'day_of_week', 'hour_of_day') are preserved without scaling,
        while other features are scaled to range (-1, 1). Original column order is maintained.
    """
    df = pd.read_csv(csv_file)
    
    time_cols = ['month', 'day_of_week', 'hour_of_day']
    non_time_cols = [c for c in df.columns if ((c not in time_cols) and (c in feature_cols))]
    
    # Store original column order
    original_order = [col for col in df.columns if col in feature_cols]
    
    # Scale non-time features if they exist
    if non_time_cols:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[non_time_cols].astype(np.float32))
        
        # Create a temporary dataframe with scaled data
        temp_df = pd.DataFrame(scaled_data, columns=non_time_cols, index=df.index)
        
        # Combine scaled and time data while preserving order
        for col in original_order:
            if col in time_cols:
                temp_df[col] = df[col]
        
        # Reorder columns to match original
        df = temp_df[original_order]
    
    # Convert to numpy array
    data = df.to_numpy(dtype=np.float32)
    
    if data.shape[0] < window_size:
        raise ValueError("Not enough data points to form a single sequence of length window_size.")
    
    # Create sequences
    sequences = []
    for start_idx in range(0, data.shape[0] - window_size + 1, stride):
        sequences.append(data[start_idx:start_idx + window_size])
    sequences = np.array(sequences)
    # Convert directly to tensor and create DataLoader
    tensor_data = torch.tensor(sequences, dtype=torch.float32,device = "cpu")
    dataset = TensorDataset(tensor_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training and Evaluation Functions
def train_model(model, dataloader, optimizer, criterion, device, scheduler, epochs=100, val_loader = None):
    model.train()
    eval_loss_hist = []
    loss_hist = []
    train_loss = 100

    for epoch in range(epochs):
        optimizer.param_groups[0]['weight_decay'] = 0.1 * (optimizer.param_groups[0]['lr'])
        total_loss = 0.0
        for data in dataloader:
            # the data variable from the loader is a list of a single value, remove it and 
            # throw the tensor into vram
            data = data[0].to(device)
            data = data.permute(0, 2, 1).contiguous()
            data = data.repeat_interleave(10, dim=2)
            optimizer.zero_grad()
            #print(model.encoder(data).shape)
            output = model(data)
            #print(output.shape)
            #print(data.shape)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(dataloader)


        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, LR: {current_lr}')
                
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data[0].to(device)
                data = data.permute(0, 2, 1).contiguous()
                data = data.repeat_interleave(10, dim=2)
                output = model(data)
                loss = criterion(output, data)
                eval_loss += loss.item()

        eval_loss = eval_loss / len(val_loader)
        print(f"Evaluation Loss: {eval_loss}")
        
        scheduler.step(eval_loss)
        loss_hist.append(train_loss)
        eval_loss_hist.append(eval_loss)
    return loss_hist, eval_loss_hist

def training_run():
    train_loader = load_csv_for_autoencoder(csv_file = ".\\final-imbalance-data-training.csv", feature_cols = feature_cols)
    train_loader, val_loader = dataloader_split(dataloader=train_loader,val_split=0.1,shuffle = True)
    print("Data preparation complete. Ready for training!")

    # Hyperparameters    
    # for the transformer the latent apce needs to be a multiple of the number of feature
    # columns and an even number
    num_latent_dim = 32 * len(feature_cols) * 2
    print("Dimmension of latent space is ", str(num_latent_dim))
    # Initialize the model, optimizer, and loss function
    #model = TransformerAutoencoder(input_size=window_size, latent_dim=num_latent_dim, init_layer_size=len(feature_cols)).to(device)
    model = DenseAutoencoder(input_size=window_size*10, latent_dim=num_latent_dim, init_layer_size=len(feature_cols)).to(device)
    #model = UnetAutoencoder(input_size=window_size, latent_dim=num_latent_dim, init_layer_size=len(feature_cols)).to(device)
    
    initial_lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=initial_lr*0.01)
    #scheduler = StepLR(optimizer, step_size=3, gamma=0.75)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=1, threshold=0.001)

    criterion = nn.HuberLoss(delta=1)
    class ExponentialLoss(nn.Module):
        def forward(self, y_pred, y_true):
            return torch.sum(torch.exp(torch.abs(y_true - y_pred)))/len(y_pred)

    criterion = ExponentialLoss()
    criterion = nn.L1Loss()
    #criterion = nn.MSELoss()
    #criterion = nn.L1Loss()

    # Train the model
    loss_hist, eval_loss_hist = train_model(model, train_loader, optimizer, criterion, device, scheduler, epochs=150, val_loader=val_loader)
    plotting = True
    if plotting == True:
        # Ensure the lists are of the same length
        if len(loss_hist) != len(eval_loss_hist):
            raise ValueError("The two lists must be of the same length.")
        
        # X-axis will be the indices of the lists
        x_values = range(len(loss_hist))
        
        # Plot the two lists
        plt.figure(figsize=(8, 5))
        plt.plot(x_values, loss_hist, marker='o', linestyle='-', label='Loss history', color='blue')
        plt.plot(x_values, eval_loss_hist, marker='s', linestyle='--', label='eval history', color='red')
        
        # Add labels, title, and legend
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.title('Two Lists Plotted on the Same Graph')
        plt.legend()
        
        # Show the grid for better visualization
        plt.grid(True)
        
        # Display the plot
        plt.show()
    
    if type(model) is UnetAutoencoder:
        model_type_and_time = "Unet"
    elif type(model) is DenseAutoencoder:
        model_type_and_time = "Dense"
    elif type(model) is TransformerAutoencoder:
        model_type_and_time = "Transformer"
    else:
        model_type_and_time = "Other"
    model_type_and_time += "-" + str(datetime.now().strftime("%Y-%m-%d"))

    with open(f'autoencoder-{model_type_and_time}.pkl', "wb") as file:
        pickle.dump(model, file)
        print(f'Model has been saved to autoencoder-{model_type_and_time}.pkl')

def dataloader_split(dataloader, val_split=0.2, shuffle=True):
    """
    Splits a DataLoader into training and validation DataLoaders.

    Parameters:
        dataloader (DataLoader): The original DataLoader.
        val_split (float): Proportion of the dataset to be used for validation.
        shuffle (bool): Whether to shuffle the dataset before splitting.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
    """
    dataset = dataloader.dataset
    dataset_size = len(dataset)
    
    # Create indices for the dataset
    indices = list(range(dataset_size))
    
    if shuffle:
        #np.random.seed(42)  # For reproducibility
        np.random.shuffle(indices)
    
    # Compute the split index
    split_idx = int(np.floor(val_split * dataset_size))
    
    # Split indices for train and validation sets
    val_indices = indices[:split_idx]
    train_indices = indices[split_idx:]
    
    # Create Samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    # Create new DataLoaders for train and validation
    train_loader = DataLoader(dataset, batch_size=dataloader.batch_size, sampler=train_sampler, drop_last=dataloader.drop_last)
    val_loader = DataLoader(dataset, batch_size=dataloader.batch_size, sampler=val_sampler, drop_last=dataloader.drop_last)
    
    return train_loader, val_loader

if __name__ == '__main__':
    # Check what version of PyTorch is installed
    print("Torch Version: ", torch.__version__)
    # Check the current CUDA version being used
    print("CUDA Version: ", torch.version.cuda)
    # Check if CUDA is available and if so, print the device name
    print("Device name:", torch.cuda.get_device_properties("cuda").name)
    # Check if FlashAttention is available
    print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())
    # Parameters
    window_size = 60*24
    stride = 30
    batch_size = 64
    feature_cols = []
    #feature_cols += ["imbalance_regulation_state"]
    #feature_cols += ["mid_price"]
    #feature_cols += ["imbalance_take_price"]
    #feature_cols += ["imbalance_feed_price"]
    feature_cols += ["month","day_of_week","hour_of_day"]
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    training_run()
    
    with open("autoencoder-Dense-2024-12-16.pkl", "rb") as file:
        model = pickle.load(file)

    test_dataloader = load_csv_for_autoencoder(csv_file = ".\\final-imbalance-data-test.csv",feature_cols=feature_cols)
    # Example batch from the validation set
    while True:
        example_batch = next(iter(test_dataloader))[0].to("cuda")
        example_batch = example_batch.permute(0,2,1).contiguous()
        
        example_batch = example_batch.repeat_interleave(10, dim=2)

        # Encode into latent space
        
        if type(model) is UnetAutoencoder:
            _, __, ___, latent = model.encode(example_batch)
            #print(_.shape)
            #print(__.shape)
            #print(___.shape)
            print(f'latent shape {latent.shape}')
            # Decode back to original space
            reconstructed = model.decode(_, __, ___, latent)
            print(f'Model output dims {reconstructed.shape}')
        elif type(model) is DenseAutoencoder:
            latent = model.encode(example_batch)
            print(f'latent shape {latent.shape}')
            # Decode back to original space 
            reconstructed = model.decode(latent)
        elif type(model) is TransformerAutoencoder:
            latent = model.encode(example_batch)
            print(f'latent shape {latent.shape}')
            # Decode back to original space 
            reconstructed = model.decode(latent)
        else:
            latent = model.encode(example_batch)
            print(f'latent shape {latent.shape}')
            # Decode back to original space 
            reconstructed = model.decode(latent)

        print(example_batch[0])
        print(reconstructed[0])

        # Example tensors (2D arrays)
        tensor1 = example_batch[0][:, ::10].cpu().detach().numpy()  # Replace with your actual tensor data
        tensor2 = reconstructed[0][:, 5::10].cpu().detach().numpy()  # Replace with your actual tensor data

        # Create x and y coordinates based on tensor shape
        x_indices = np.arange(tensor1.shape[1])  # Assuming shape is (rows, cols)
        y_indices = np.arange(tensor1.shape[0])

        # Create a figure with subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Tensor 1", "Tensor 2"))

        # Add first tensor as a heatmap
        # Determine global min and max
        zmin = np.min(tensor1)
        zmax = np.max(tensor2)
        zmin = np.min([zmin, -1.0])
        zmax = np.max([zmax, 1.0])

        # Create the figure

        # Add first tensor as a heatmap
        fig.add_trace(
            go.Heatmap(z=tensor1, x=x_indices, y=y_indices, colorscale='Turbo', zmin=zmin,zmax=zmax),
            row=1, col=1
        )

        # Add second tensor as a heatmap
        fig.add_trace(
            go.Heatmap(z=tensor2, x=x_indices, y=y_indices, colorscale='Turbo', zmin=zmin, zmax=zmax),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(title_text="Comparison of Two Tensors", height=600, width=800)

        # Show the figure
        fig.show()
        input()

import torch
import torch.nn as nn
import torch.optim as optim

class Conv1DAutoencoder(nn.Module):
    def __init__(self, input_size, latent_dim):
        """
        Initialize the Conv1D Autoencoder.

        Args:
            input_size (int): Size of the input sequence (timesteps).
            latent_dim (int): Dimensionality of the latent space.
        """
        super(Conv1DAutoencoder, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose1d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Perform a full forward pass (encode and decode)."""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x):
        """Encode input data into the latent space."""
        return self.encoder(x)

    def decode(self, latent):
        """Decode latent representation back to the original space."""
        return self.decoder(latent)

# Training and Evaluation Functions
def train_model(model, dataloader, optimizer, criterion, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data)
            total_loss += loss.item()
    print(f"Evaluation Loss: {total_loss / len(dataloader)}")
    return total_loss / len(dataloader)


import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Example input data
timesteps = 86400
x_train = np.random.rand(100, timesteps, 1).astype(np.float32)
x_val = np.random.rand(20, timesteps, 1).astype(np.float32)

# Convert to PyTorch tensors
train_dataset = TensorDataset(torch.tensor(x_train))
val_dataset = TensorDataset(torch.tensor(x_val))

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
# Hyperparameters
latent_dim = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model, optimizer, and loss function
model = Conv1DAutoencoder(input_size=timesteps, latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Train the model
train_model(model, train_loader, optimizer, criterion, device, epochs=10)

# Evaluate the model
evaluate_model(model, val_loader, criterion, device)

# Example batch from the validation set
example_batch = next(iter(val_loader))[0].to(device)

# Encode into latent space
latent = model.encode(example_batch)

# Decode back to original space
reconstructed = model.decode(latent)

# Full forward pass
output = model(example_batch)

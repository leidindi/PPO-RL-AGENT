import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# Check if CUDA is available, else fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'We are using:{device}')

# Generate a synthetic dataset (non-linearly separable)
def generate_spiral_data(n_samples=1000, noise=0.2):
    X, y = make_moons(n_samples=n_samples, noise=noise)
    return X, y

# Create the dataset
X, y = generate_spiral_data(n_samples=100000)
print("data made")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# Define a simple feedforward neural network with at least 2 hidden layers
class FeedforwardNN(nn.Module):
    def __init__(self):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # Input layer (2 features)
        self.fc2 = nn.Linear(64, 128) # Hidden layer 1
        self.fc3 = nn.Linear(128, 64) # Hidden layer 2
        self.fc4 = nn.Linear(64, 128) # Hidden layer 1
        self.fc5 = nn.Linear(128, 64) # Hidden layer 2
        self.out = nn.Linear(64, 2)  # Output layer (2 classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.out(x)
        return x

# Initialize the model, define loss function and optimizer
model = FeedforwardNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
n_epochs = 100000
for epoch in range(n_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % n_epochs/10 == 0:
        print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')

# Test the model on the test set
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs.data, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Visualize decision boundary with Plotly
def plot_decision_boundary_plotly(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)
    
    # Forward pass through the model to get predictions
    with torch.no_grad():
        Z = model(grid_tensor)
        _, Z = torch.max(Z, 1)
    Z = Z.cpu().numpy().reshape(xx.shape)

    # Create the Plotly figure
    fig = go.Figure()

    # Add decision boundary as a contour plot
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, 0.01),
        y=np.arange(y_min, y_max, 0.01),
        z=Z,
        colorscale='RdBu',
        opacity=0.5
    ))

    # Add scatter plot for the dataset
    fig.add_trace(go.Scatter(
        x=X[:, 0], y=X[:, 1],
        mode='markers',
        marker=dict(color=y, colorscale='Viridis', line=dict(width=1)),
        showlegend=False
    ))

    fig.update_layout(title='Decision Boundary of FFNN',
                      xaxis_title='Feature 1',
                      yaxis_title='Feature 2')

    fig.show()

# Plot decision boundary
X_np = X_test.cpu().numpy()
y_np = y_test.cpu().numpy()
plot_decision_boundary_plotly(model, X_np, y_np)

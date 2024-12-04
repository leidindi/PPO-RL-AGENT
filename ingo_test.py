import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from kan import *
from kan.KANLayer import *
import gc
import sys
def NNtest():
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
            self.fc1 = nn.Linear(2, 4)  # Input layer (2 features)
            self.fc2 = nn.Linear(4, 8) # Hidden layer 1
            self.fc3 = nn.Linear(8, 4) # Hidden layer 2
            self.KAN1 = KANLayer(in_dim=2, out_dim=4)
            self.KAN2 = KANLayer(in_dim=4, out_dim=8)
            self.KAN3 = KANLayer(in_dim=8, out_dim=4)
            self.KAN4 = KANLayer(in_dim=4, out_dim=8)
            self.KAN5 = KANLayer(in_dim=8, out_dim=4)
            self.KAN6 = KANLayer(in_dim=4, out_dim=2)
            self.fc4 = nn.Linear(4, 8) # Hidden layer 2
            self.fc5 = nn.Linear(8, 4) # Hidden layer 2
            self.out = nn.Linear(4, 2)  # Output layer (2 classes)

        def forward(self, x):
            #x = torch.relu(self.fc1(x))
            #x = torch.relu(self.fc2(x))
            #x = torch.relu(self.fc3(x))
            #x = torch.relu(self.fc4(x))
            #x = torch.relu(self.fc5(x))
            #x = self.out(x)

            
            x, _, __, ___ = self.KAN1.forward(x)
            x, _, __, ___ = self.KAN6.forward(x)
            #x, _, __, ___ = self.KAN3.forward(x)
            #x, _, __, ___ = self.KAN4.forward(x)
            #x, _, __, ___ = self.KAN5.forward(x)
            #x, _, __, ___ = self.KAN6.forward(x)

            return x

    # Initialize the model, define loss function and optimizer
    model = FeedforwardNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    n_epochs = 1000
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

def KANtest():
    # check memory if its not emptying
    torch.set_default_dtype(torch.float32)

    path1 = r"C:\Users\ingoa\Documents\GitHub\simpl-energy-KAN-do-it\ingo fikt\ElectricDevices_TRAIN.txt"
    path2 = r"C:\Users\ingoa\Documents\GitHub\simpl-energy-KAN-do-it\ingo fikt\ElectricDevices_TEST.txt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        torch.set_default_device(device)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)



    def load_data(filepath):
        """
        Helper function to load data from a text file.
        Assumes each line of the file is a time series row.
        """
        with open(filepath, 'r') as file:
            data = file.readlines()
        return data

    def split_and_combine_data(path1, path2, split_index=6047):
        # Load data from both paths
        data1 = load_data(path1)
        data2 = load_data(path2)

        # Split data from path2
        path2_train_data = data2[:split_index]  # First 6047 lines for training
        path2_test_data = data2[split_index:]   # Remaining lines for testing

        # Combine path1 data with first part of path2 data for training
        training_data = data1 + path2_train_data

        return training_data, path2_test_data

    def prepare_datasets(path1, path2, split_index=6047):
        # Split and combine the data
        training_data, test_data = split_and_combine_data(path1, path2, split_index)

        # Convert data into suitable format for kan.utils.create_dataset
        # Assuming the first column is the label and the rest is the time series data
        train_labels = []
        train_series = []
        test_labels = []
        test_series = []

        # Process training data
        for line in training_data:
            values = line.strip().split()  # Split by spaces
            train_labels.append(float(values[0]))  # First column is the label
            train_series.append([float(x) for x in values[1:]])  # Rest are time series data

        # Process test data
        for line in test_data:
            values = line.strip().split()
            test_labels.append(float(values[0]))
            test_series.append([float(x) for x in values[1:]])

        return torch.tensor(train_series), torch.tensor(train_labels), torch.tensor(test_series), torch.tensor(test_labels)


    # Prepare the datasets

    dataset = {}
    dataset['train_input'], dataset['train_label'], dataset['test_input'], dataset['test_label'] = prepare_datasets(path1, path2)
    dataset['train_label'] = dataset['train_label'].reshape(dataset['train_label'].shape[0], 1) - 1
    dataset['test_label'] = dataset['test_label'].reshape(dataset['test_label'].shape[0], 1) - 1
    all_labels_original = copy.deepcopy(torch.cat((dataset['train_label'],dataset['test_label']),dim=0).flatten().tolist())
    def compress_list_by_average(data, factor=8):
        compressed_data = []
        
        # Iterate over each sub-list in the data
        for sublist in data:
            # Reshape the sublist into (12, 8) shape and compute the mean along axis 1
            if isinstance(sublist, np.float64): 
                    break
            
            compressed_sublist = []
            for i in range(0, len(sublist), factor):
                
                sub = sublist[i:i+factor]
                compressed_sublist.append(np.mean(sub.detach().numpy()))
            compressed_data.append(compressed_sublist)
            
        return compressed_data

    #All Factors of 96:
    #1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 96
    # Compress each sub-list by a factor of 8 (using average)
    data_factoring = 4
    dataset['train_input'] = torch.tensor(compress_list_by_average(dataset['train_input'].cpu(), factor=data_factoring))
    dataset['test_input'] = torch.tensor(compress_list_by_average(dataset['test_input'].cpu(), factor=data_factoring))
    dataset['train_input'].to(device)
    dataset['train_label'].to(device)
    dataset['test_input'].to(device)
    dataset['test_label'].to(device)
    print(device)
    pass
    # create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
    # input data normalization step
    mean = dataset['train_input'].mean()
    std = dataset['train_input'].std()

    # Normalize the tensor
    dataset['train_input'] = (dataset['train_input'] - mean) / std
    dataset['test_input'] = (dataset['test_input'] - mean) / std
    # label data one hot encoding step
    # Number of classes for one-hot encoding (in this case, 8)
    num_classes = 7

    # Create one-hot encoding using torch.nn.functional

    dataset['train_label'] = dataset['train_label'].float()
    dataset['test_label'] = dataset['test_label'].float()

    dataset['train_label'] = torch.flatten(dataset['train_label'], start_dim=1)
    dataset['test_label'] = torch.flatten(dataset['test_label'], start_dim=1)
    # plot KAN at initialization
    model = KAN(width=[dataset['train_input'].shape[1],3,3,3,7], grid=5, k=3, seed=1997, device=device)
    print(f'using : {device}')
    model.to(device)
    model(dataset['train_input']);
    model.plot(scale=4.0)#,sample=True)
    # train the model
    model.save_act=True

    print(dataset['train_input'].shape)
    print(dataset['train_label'].shape)
    print(dataset['test_input'].shape)
    print(dataset['test_label'].shape)

    # Get the unique labels and their counts
    #print(dataset['train_label'][745:899])
    labels, counts = torch.unique(dataset['train_label'], return_counts=True)
    #print(labels)
    #print(counts)
    # Calculate class weights (inverse of the counts)
    weights = 1.0 / counts.float()

    # Normalize weights so that the sum equals the number of classes
    weights = weights / weights.sum() * len(weights)
    #print(weights)
    
    dataset['train_label'] = torch.nn.functional.one_hot(dataset['train_label'].long(), num_classes=num_classes)
    dataset['test_label'] = torch.nn.functional.one_hot(dataset['test_label'].long(), num_classes=num_classes)
    dataset['train_label'] = dataset['train_label'].squeeze(1).float()
    dataset['test_label'] = dataset['test_label'].squeeze(1).float()
    # Create CrossEntropyLoss with class weights
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))

    results = model.fit(dataset, opt="Adam", steps=1000, lamb=0.001,lr = 0.01, batch=dataset['test_label'].shape[0], loss_fn=criterion)
    
    import plotly.graph_objects as go
    # Get the values from the dictionary
    train_loss = results['train_loss']
    test_loss = results['test_loss']

    # Create a figure
    fig = go.Figure()

    # Plot training loss
    fig.add_trace(go.Scatter(
        x=list(range(len(train_loss))),
        y=train_loss,
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='blue'),
        marker=dict(symbol='circle')
    ))

    # Plot test loss
    fig.add_trace(go.Scatter(
        x=list(range(len(test_loss))),
        y=test_loss,
        mode='lines+markers',
        name='Test Loss',
        line=dict(color='green', dash='dash'),
        marker=dict(symbol='x')
    ))

    # Add labels and title
    fig.update_layout(
        title='Training Loss and Test Loss',
        xaxis_title='Iterations',
        yaxis_title='Loss',
        legend_title='Legend',
        template='plotly_white',
        showlegend=True
    )

    # Add grid
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    # Show the plot
    fig.show()

    correct = 0
    import time
    model_guess = model(dataset['test_input'])
    for index, label in enumerate(dataset["test_label"]):
        guess = model_guess[index]
        if model_guess.shape[1] == 1:
            guess = int(np.round(guess.item()))
            correct_answer = int(np.round(label.item()))
        else:
            guess = torch.argmax(guess)
            correct_answer = torch.argmax(label)
        if guess == correct_answer:
            correct += 1
        sys.stdout.write(f'\rAccuracy of model predictions on the test set : {correct/(index+1):.2}, Model Guess : {guess}, actual answer {correct_answer}')
        sys.stdout.flush() 
    model = model.prune()
    results = model.fit(dataset, opt="Adam", steps=2000, lamb=0.0001,lr = 0.001, batch=dataset['test_label'].shape[0], loss_fn=criterion)
    import plotly.graph_objects as go
    # Get the values from the dictionary
    train_loss = results['train_loss']
    test_loss = results['test_loss']

    # Create a figure
    fig = go.Figure()

    # Plot training loss
    fig.add_trace(go.Scatter(
        x=list(range(len(train_loss))),
        y=train_loss,
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='blue'),
        marker=dict(symbol='circle')
    ))

    # Plot test loss
    fig.add_trace(go.Scatter(
        x=list(range(len(test_loss))),
        y=test_loss,
        mode='lines+markers',
        name='Test Loss',
        line=dict(color='green', dash='dash'),
        marker=dict(symbol='x')
    ))

    # Add labels and title
    fig.update_layout(
        title='Training Loss and Test Loss',
        xaxis_title='Iterations',
        yaxis_title='Loss',
        legend_title='Legend',
        template='plotly_white',
        showlegend=True
    )

    # Add grid
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    # Show the plot
    fig.show()
    correct = 0
    import time
    model_guess = model(dataset['test_input'])
    for index, label in enumerate(dataset["test_label"]):
        guess = model_guess[index]
        if model_guess.shape[1] == 1:
            guess = int(np.round(guess.item()))
            correct_answer = int(np.round(label.item()))
        else:
            guess = torch.argmax(guess)
            correct_answer = torch.argmax(label)
        if guess == correct_answer:
            correct += 1
        sys.stdout.write(f'\rAccuracy of model predictions on the test set : {correct/(index+1):.2}, Model Guess : {guess}, actual answer {correct_answer}')
        sys.stdout.flush() 
    model = model.prune()
    results = model.fit(dataset, opt="LBFGS", steps=1000, lamb=0.001,lr = 0.01, batch=dataset['test_label'].shape[0], loss_fn=criterion)
    import plotly.graph_objects as go
    # Get the values from the dictionary
    train_loss = results['train_loss']
    test_loss = results['test_loss']

    # Create a figure
    fig = go.Figure()

    # Plot training loss
    fig.add_trace(go.Scatter(
        x=list(range(len(train_loss))),
        y=train_loss,
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='blue'),
        marker=dict(symbol='circle')
    ))

    # Plot test loss
    fig.add_trace(go.Scatter(
        x=list(range(len(test_loss))),
        y=test_loss,
        mode='lines+markers',
        name='Test Loss',
        line=dict(color='green', dash='dash'),
        marker=dict(symbol='x')
    ))

    # Add labels and title
    fig.update_layout(
        title='Training Loss and Test Loss',
        xaxis_title='Iterations',
        yaxis_title='Loss',
        legend_title='Legend',
        template='plotly_white',
        showlegend=True
    )

    # Add grid
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    # Show the plot
    fig.show()
    correct = 0
    import time
    model_guess = model(dataset['test_input'])
    for index, label in enumerate(dataset["test_label"]):
        guess = model_guess[index]
        if model_guess.shape[1] == 1:
            guess = int(np.round(guess.item()))
            correct_answer = int(np.round(label.item()))
        else:
            guess = torch.argmax(guess)
            correct_answer = torch.argmax(label)
        if guess == correct_answer:
            correct += 1
        sys.stdout.write(f'\rAccuracy of model predictions on the test set : {correct/(index+1):.2}, Model Guess : {guess}, actual answer {correct_answer}')
        sys.stdout.flush() 

import copy
reward_arr = np.random.uniform(-1, 0, 10000)
reward_arr -= 1
reward_arr[-1] = 100
dones_arr=np.zeros_like(reward_arr)
dones_arr[-1] = 1
values1=np.random.uniform(-25, 25, 10000)
values=copy.deepcopy(values1)

def deltatest1():
    global values1
    values = copy.deepcopy(values1)
    gamma = 0.99
    gae_lambda = 0.95
    advantage = np.zeros_like(reward_arr)
    values = np.append(values, values[-1], axis=None)

    deltas = reward_arr + gamma * values[1:] * (1 - dones_arr) - values[:-1]
    gae = 0
    for t in reversed(range(len(deltas))):
        gae = deltas[t] + gamma * gae_lambda * (1 - dones_arr[t]) * gae
        advantage[t] = gae
    return advantage

def deltatest2():
    global values
    gamma = 0.99
    gae_lambda = 0.95
    advantage = np.zeros_like(reward_arr)
    for t in range(len(reward_arr)-1):
        discount = 1
        a_t = 0
        for k in range(t, len(reward_arr)-1):

            done_switch = 1-int(dones_arr[k])
            rewards_diff = gamma*values[k+1]*done_switch - values[k]
            all_rewards = reward_arr[k] + rewards_diff
            a_t += discount*all_rewards
            discount *= gamma*gae_lambda
        advantage[t] = a_t
    return advantage

def timer():
    import time

    start_time = time.time()
    result1 = deltatest1()
    end_time = time.time()

    # Print the execution time
    print(f"Execution time: {end_time - start_time:.6f} seconds")


    start_time = time.time()
    result2 = deltatest2()
    end_time = time.time()

    # Print the execution time
    print(f"Execution time: {end_time - start_time:.6f} seconds")
    print(result1-result2)

    from matplotlib import pyplot
    pyplot.plot(result1)
    pyplot.show()
    pyplot.plot(result2)
    pyplot.show()

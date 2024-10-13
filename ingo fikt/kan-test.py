import torch
import os
# create dataset f(x,y) = exp(sin(pi*x)+y^2)
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)

path2 = r"ElectricDevices_TEST.txt"
path1 = r"ElectricDevices_TRAIN.txt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

import numpy as np
from kan.utils import create_dataset

import numpy as np
from kan.utils import create_dataset

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
dataset['train_input'],dataset['train_output'],dataset['test_input'],dataset['test_output'] = prepare_datasets(path1, path2)

# Now train_dataset and test_dataset are ready to be used with kan.utils


#dataset = create_dataset(f, n_var=2, device=device)
#dataset['train_input']
#x = torch.zeros(2, 1000, dtype=torch.float64)
#x = x.T

print(torch.mean(dataset['train_input']))
print(dataset["train_input"].shape)
dataset['train_input'] = dataset['train_input'] #+ (0.01)*torch.randn(*dataset["train_input"].shape)
print(torch.mean(dataset['train_input']))
dataset['test_input'] = dataset['test_input'] #+ (0.01)*torch.randn(*dataset["test_input"].shape)

print("dfa")
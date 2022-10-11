import logging
import torch
import math
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch.nn as nn
import torch.nn.functional as F
import plotly.express as px
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)
torch.manual_seed(13)


class PriceNightDataset(Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y), "Data and labels must be of equal length."
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y)


class FeedforwardNeuralNetModel(nn.Module):
    # Initialize the layers
    def __init__(self, input_dim, hidden_dim_array, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dim_array)):
            self.layers.append(nn.Linear(input_dim, hidden_dim_array[i]))
            input_dim = hidden_dim_array[i]  # For the next layer
            self.layers.append(nn.ReLU())  # Activation function
        self.layers.append(nn.Linear(input_dim, output_dim))
        print(self.layers)
    
    # Perform the computation
    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output


def split_dataset(dataset, targets, random_state):
    """Splits up the dataset into training, validation and testing datasets
    in the repective ratio of 60:20:20.
    Args:
        dataset (DataFrame): The dataset to be split.
        targets (list): A list of columns to be used as the targets.
        random_state (int): The random state used in the split.
    Returns:
        (tuple): (X_train, y_train, X_validation, y_validation, X_test, y_test)
            in tensor form.
    """
    X = torch.tensor(dataset.drop(targets, axis=1).values).float()
    y = torch.tensor(dataset[targets].values).float()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)
    return (X_train, y_train, X_validation, y_validation, X_test, y_test)


def create_dataloader(X_train, y_train, batch_size):
    """Creates a DataLoader from the training dataset.
    Args:
        X_train (tensor): The tensor containing the training features.
        y_train (tensor): The tensor containing the training targets.
        batch_size (int: The size of one batch.
    Returns:
        dataloader (class): DataLoader created from the training set.
    """
    train_dataset = PriceNightDataset(X_train, y_train)
    dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def get_num_epochs(n_iters, batch_size, X_train):
    """Calculate number of passes through the entire training set.
    Args:
        n_iters (int): The number of batches iterated over.
        batch_size (int): The size of one batch.
        X_train (tensor): The tensor containing the training features.
    Returns:
        (int): The number of passes through the entire dataset.
    """
    return int(n_iters / (len(X_train) / batch_size))


def get_nn_config(path):
    """Load config yaml file.
    Args:
        path (str): Path to config file.
    """
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.Loader)
    return config


def calculate_regression_metrics(y_train, y_train_pred, y_validation, y_validation_pred, y_test, y_test_pred):
    """Calculates the RMSE and R2 score of a regression model.
    Args:
        y_train (array): Features for training.
        y_train_pred (array): Features predicted with training set.
        y_validation (array): Features for validation
        y_validation_pred (array): Features predicted with validation set.
        y_test (array): Features for testing.
        y_test_pred(array): Features predicted with testing set.
    Returns:
        metrics (dict): Training, validation and testing performance metrics.
    """
    rmse_train = math.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_train = r2_score(y_train, y_train_pred)
    rmse_validation = math.sqrt(mean_squared_error(y_validation, y_validation_pred))
    r2_validation = r2_score(y_validation, y_validation_pred)
    rmse_test = math.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_test = r2_score(y_test, y_test_pred)
    metrics = {
        'Training RMSE': rmse_train,
        'Training R2 score': r2_train,
        'Validation RMSE': rmse_validation,
        'Validation R2 score': r2_validation,
        'Test RMSE': rmse_test,
        'Test R2 score': r2_test
        }
    return metrics

# writer = SummaryWriter('project/models/neural_networks/runs')

def train_model(sets, dataloader, config, num_epochs):
    """Trains the feed forward neural network.
    Args:
        dataloader (class): DataLoader created from the training set.
        config (dict): Network configuration settings.
        num_epochs (int): The number of passes through the entire dataset.
    """
    input_dim = sets[0].shape[1]
    output_dim = sets[1].shape[1]
    model = FeedforwardNeuralNetModel(input_dim, config['hidden_dim_array'], output_dim)
    criterion = nn.MSELoss()
    opt = config['optimiser'](model.parameters(), lr=config['learning_rate'])

    for i in range(num_epochs):
        for x_train, y_train in dataloader:
            opt.zero_grad()
            pred = model(x_train)
            loss = criterion(pred, y_train)
            loss.backward()
            opt.step()
            opt.zero_grad()
        # writer.add_scalar('RMSE/train', math.sqrt(criterion(model(features), targets).item()), i)
        if i % 50 == 0 or i == range(num_epochs)[-1]:
            logging.info(f'Epoch {i} training loss: {criterion(model(sets[0]), sets[1])}')


def create_and_train_nn():
    """Loads the dataset in the form of a Pandas DataFrame, converts
    to train, validation and testing tensors and then trains the
    feed forward model. Saves the hyperparameters, model and metrics.
    """
    numerical_dataset = pd.read_csv('project/dataframes/numerical_data.csv', index_col=0)
    sets = split_dataset(numerical_dataset, ['price_night'], random_state=13)
    # num_epochs = get_num_epochs(3000, 100, sets[0])
    # dataloader = create_dataloader(sets[0], sets[1], batch_size=100)
    # config = get_nn_config('project/models/neural_networks/regression/nn_config.yaml')
    # train_model(sets, dataloader, config, num_epochs)
    # writer.flush()
    # writer.close()

if __name__ == '__main__':
    create_and_train_nn()

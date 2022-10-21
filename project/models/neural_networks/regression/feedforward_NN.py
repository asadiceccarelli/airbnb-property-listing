import logging
import os
import glob
import math
import torch
import json
import yaml
import random
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional import mean_squared_error, r2_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from time import time

logging.basicConfig(level=logging.INFO)

torch.manual_seed(13)
seed = 13
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class PriceNightDataset(Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y), 'Data and labels must be of equal length.'
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
        self.hidden_dim_array = hidden_dim_array
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dim_array)):
            self.layers.append(nn.Linear(input_dim, hidden_dim_array[i]))
            input_dim = hidden_dim_array[i]  # For the next layer
            self.layers.append(nn.ReLU())  # Activation function
        self.layers.append(nn.Linear(input_dim, output_dim))
        self.weights_init()

    def weights_init(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(layer.bias, 1)
    
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


def calculate_regression_metrics(training_time, inference_latency, y_train, y_train_pred, y_validation, y_validation_pred, y_test, y_test_pred):
    """Calculates the RMSE and R2 score of a regression model.
    Args:
        training_time (float): Number of seconds to train model.
        y_train (array): Features for training.
        y_train_pred (array): Features predicted with training set.
        y_validation (array): Features for validation
        y_validation_pred (array): Features predicted with validation set.
        y_test (array): Features for testing.
        y_test_pred(array): Features predicted with testing set.
    Returns:
        metrics (dict): Training, validation and testing performance metrics.
    """
    rmse_train = mean_squared_error(y_train_pred, y_train, squared=False).item()
    r2_train = r2_score(y_train_pred, y_train).item()
    rmse_validation = mean_squared_error(y_validation_pred, y_validation, squared=False).item()
    r2_validation = r2_score(y_validation_pred, y_validation).item()
    rmse_test = mean_squared_error(y_test_pred, y_test, squared=False).item()
    r2_test = r2_score(y_test_pred, y_test).item()
    metrics = {
        'Training time': training_time,
        'Inference latency': inference_latency,
        'Training RMSE': rmse_train,
        'Training R2 score': r2_train,
        'Validation RMSE': rmse_validation,
        'Validation R2 score': r2_validation,
        'Test RMSE': rmse_test,
        'Test R2 score': r2_test
        }
    return metrics


def train_model(X_train, y_train, dataloader, config, num_epochs):
    """Trains the feed forward neural network.
    Args:
        X_train (tensor): The tensor containing the training features.
        y_train (tensor): The tensor containing the training targets.
        dataloader (class): DataLoader created from the training set.
        config (dict): Network configuration settings.
        num_epochs (int): The number of passes through the entire dataset.
    Returns:
        model (class): The trained model.
        opt (class): The optimiser used to update parameters based on
            computed gradients.
        training_time (float): Number of seconds taken to train the model.
        mean_inference_latency (float): The average number of seconds to
            make a prediction.
    """
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = FeedforwardNeuralNetModel(input_dim, config['hidden_dim_array'], output_dim)
    criterion = nn.MSELoss()
    opt = config['optimiser'](model.parameters(), lr=config['learning_rate'])
    writer = SummaryWriter('project/models/neural_networks/regression/runs')

    start_time = time()
    mean_inference_latency = 0
    counter = 0
    for i in range(num_epochs):
        for X, y in dataloader:
            opt.zero_grad()
            inference_latency_start = time()
            pred = model(X)
            inference_latency = time() - inference_latency_start
            counter += 1
            mean_inference_latency += (inference_latency - mean_inference_latency) / counter  # Incremental mean
            loss = criterion(pred, y)
            loss.backward()
            opt.step()
            writer.add_scalar('RMSE/train', math.sqrt(criterion(model(X_train), y_train).item()), i)
        if i % 100 == 0 or i == range(num_epochs)[-1]:
            logging.info(f'Epoch {i} MSE training loss: {criterion(model(X_train), y_train)}')
    writer.flush()
    writer.close()
    end_time = time()
    training_time = end_time - start_time
    return model, opt, training_time, mean_inference_latency


def save_model(model, opt, sets, training_time, mean_inference_latency):
    """Saves the model as a .pt file, the hyperparameters as a .yml file
    and the metrics as a .json file.
    Args:
        model (class): The PyTorch model used.
        opt (class): The optimiser used to update parameters based on
            computed gradients.
        sets (tuple): (X_train, y_train, X_validation, y_validation, X_test, y_test)
        training_time (float): Number of seconds taken to train the model.
        mean_inference_latency (float): The average number of seconds to
            make a prediction.
    """
    y_train_pred = model(sets[0])
    y_validation_pred = model(sets[2])
    y_test_pred = model(sets[4])
    hyperparameters = {
        'optimiser': torch.optim.SGD,
        'learning_rate': opt.__dict__['defaults']['lr'],
        'hidden_dim_array': model.hidden_dim_array
        }
    metrics_dict = calculate_regression_metrics(
        training_time,
        mean_inference_latency,
        sets[1], y_train_pred,
        sets[3], y_validation_pred,
        sets[5], y_test_pred
    )
    date_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    os.mkdir(f'project/models/neural_networks/regression/trains/{date_time}')
    torch.save(model, f'project/models/neural_networks/regression/trains/{date_time}/model.pt')
    with open(f'project/models/neural_networks/regression/trains/{date_time}/hyperparameters.yml', 'w') as outfile:
        yaml.dump(hyperparameters, outfile)
    with open(f'project/models/neural_networks/regression/trains/{date_time}/metrics.json', 'w') as outfile:
        json.dump(metrics_dict, outfile)


def create_and_train_nn(config):
    """Loads the dataset in the form of a Pandas DataFrame, converts
    to train, validation and testing tensors and then trains the
    feed forward model. Saves the hyperparameters, model and metrics.
    Args:
        config (dict): Containing information on the optimiser, learning
            rate and hidden layer depth and width.
    """
    torch.manual_seed(13)
    numerical_dataset = pd.read_csv('project/data/structured/numerical_data.csv', index_col=0)
    sets = split_dataset(numerical_dataset, ['price_night'], random_state=13)
    num_epochs = get_num_epochs(30000, 100, sets[0])
    dataloader = create_dataloader(sets[0], sets[1], batch_size=100)
    model, opt, training_time, mean_inference_latency = train_model(sets[0], sets[1], dataloader, config, num_epochs)
    save_model(model, opt, sets, training_time, mean_inference_latency)


def generate_nn_config():
    """Creates multiple configuration dictionaries containing info on 
    the optimiser, learning rate and depth and width of the hidden
    layers.
    Returns:
        config_dict (dict): Containing multiple config dictionaries.
    """
    learning_rate_tests = [1e-4, 1e-5, 1e-6]
    hidden_dim_array_tests = [
        [2], [4], [6], [8], [10],
        [2, 2], [4, 4], [6, 6], [8, 8], [10, 10],
        [2, 2, 2], [4, 4, 4], [6, 6, 6], [8, 8, 8], [10, 10, 10],
        [8, 4], [8, 6], [8, 6], [6, 4], [6, 2], [4, 2],
        [8, 6, 4], [10, 6, 4], [10, 6, 2], [8, 4, 2], [6, 4, 2],
        [10, 8, 6, 4],
        [10, 8, 6, 4, 2],
    ]
    config_dict = {}
    counter = 0
    for learning_rate in learning_rate_tests:
        for hidden_dim_array in hidden_dim_array_tests:
            config_dict[counter] = {}
            config_dict[counter]['optimiser'] = torch.optim.SGD
            config_dict[counter]['learning_rate'] = learning_rate
            config_dict[counter]['hidden_dim_array'] = hidden_dim_array
            counter += 1
    return config_dict


def find_best_nn(config_dict):
    """Creates and trains a new feed forward NN model for each
    configuration and selects the best model based on validation
    set RMSE. Outputs best model's hyperparameters to log.
    Args:
        config_dict (dict): Containing multiple config dictionaries.
    """
    for i in config_dict:
        logging.info(f'Test {i}\nUsing parameters: {config_dict[i]}.')
        create_and_train_nn(config_dict[i])
    paths = glob.glob('project/models/neural_networks/regression/trains/*')
    validation_rmse_dict = {}
    for path in paths:
        name = path.split('/')[-1]
        with open(f'{path}/metrics.json', 'r') as file:
            validation_rmse_dict[name] = json.load(file)['Validation RMSE']
    best_model = min(validation_rmse_dict, key=validation_rmse_dict.get)
    with open(f'project/models/neural_networks/regression/trains/{best_model}/hyperparameters.yml', 'r') as file:
        best_params = yaml.load(file, Loader=yaml.Loader)
    logging.info(f'Best model: {best_model}\nBest hyperparameters: {best_params}')


def train_model_once():
    with open('project/models/neural_networks/regression/trains/2022-10-21_10:31:14/hyperparameters.yml', 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)
    create_and_train_nn(config)


if __name__ == '__main__':
    # find_best_nn(generate_nn_config())
    train_model_once()

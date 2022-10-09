import logging
import torch
import math
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import plotly.express as px
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)
torch.manual_seed(13)

#################################
# 1. DATASET AND MODEL CLASS
#################################

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
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU() # Activation function
        self.linear2 = nn.Linear(hidden_dim, output_dim)
    
    # Perform the computation
    def forward(self, x):
        output = self.linear1(x) 
        output = self.act1(output)
        output = self.linear2(output)
        return output


###################################
# 2. CREATE DATALOADER
###################################

def get_num_epochs(n_iters, batch_size, features):
    return int(n_iters / (len(features) / batch_size))

def create_dataloader(features, targets, batch_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=13)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25, random_state=13)
    train_dataset = PriceNightDataset(X_train, y_train)
    validation_dataset = PriceNightDataset(X_validation, y_validation)
    test_dataset = PriceNightDataset(X_test, y_test)
    dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


###################################
# 4. GET MODEL PARAMETERS
###################################

def get_nn_config():
        with open('project/models/neural_network/nn_config.yaml') as file:
            config = yaml.load(file, Loader=yaml.Loader)
        return config


###################################
# 4. TRAIN MODEL
###################################

# writer = SummaryWriter('project/models/neural_networks/runs')

def train_model(dataloader, config, num_epochs):
    input_dim = 11
    output_dim = 1
    model = FeedforwardNeuralNetModel(input_dim, config['hidden_width_layer'], output_dim)

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
        writer.add_scalar('RMSE/train', math.sqrt(criterion(model(features), targets).item()), i)
        if i % 50 == 0 or i == range(num_epochs)[-1]:
            logging.info(f'Epoch {i} training loss: {criterion(model(features), targets)}')


def create_and_train_nn():
    numerical_dataset = pd.read_csv('project/dataframes/numerical_data.csv', index_col=0)
    features = torch.tensor(numerical_dataset.drop('price_night', axis=1).values).float()  # 890x11
    targets = torch.tensor(numerical_dataset['price_night']).float()  # 890x1
    num_epochs = get_num_epochs(3000, 100, features)
    dataloader = create_dataloader(features, targets, batch_size=100, random_state=13)
    config = get_nn_config()
    train_model(config, num_epochs)
    writer.flush()
    writer.close()

if __name__ == '__main__':
    create_and_train_nn()


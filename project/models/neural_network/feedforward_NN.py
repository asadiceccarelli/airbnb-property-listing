import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import plotly.express as px
from torch.utils.data import Dataset, DataLoader


##############################
# 1. LOAD THE DATASET
##############################

torch.manual_seed(13)

numerical_dataset = pd.read_csv('project/dataframes/numerical_data.csv', index_col=0)
features = torch.tensor(numerical_dataset.drop('price_night', axis=1).values).float()  # 890x11
targets = torch.tensor(numerical_dataset['price_night']).float()  # 890x1

################################
# 2. MAKE THE DATASET ITERABLE
################################

class PriceNightDataset(Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y), "Data and labels must be of equal length."
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y)

batch_size = 100
# n_iters = 3000
# num_epochs = n_iters / (len(features) / batch_size)
# num_epochs = int(num_epochs)
num_epochs = 301

dataset = PriceNightDataset(features, targets)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

################################
# 3. CREATE MODEL CLASS
################################

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

################################
# 4. INSTANTIATE MODEL CLASS
################################

input_dim = 11
hidden_dim = 8
output_dim = 1

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

################################
# 5. SET PARAMETERS
################################

criterion = nn.MSELoss()

learning_rate = 1e-5
opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

################################
# 5. TRAIN THE MODEL
################################

for i in range(num_epochs):
    for x_train, y_train in dataloader:
        opt.zero_grad()
        pred = model(x_train)
        loss = criterion(pred, y_train)
        loss.backward()
        opt.step()
    if i % 50 == 0:
        print(f'Epoch {i} training loss: {criterion(model(features), targets)}')

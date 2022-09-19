import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Convert inputs and targets to tensors
numerical_dataset = pd.read_csv('project/dataframes/numerical_data.csv', index_col=0)
inputs = torch.tensor(numerical_dataset.drop('price_night', axis=1).values).float()  # 890x11
features = torch.tensor(numerical_dataset['price_night']).float()  # 890x1

class price_night_Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y), "Data and labels must be of equal length."
        self.X = X
        self.y = y

    # Not dependent on index
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y)


# Define dataset
dataset = price_night_Dataset(inputs, features)
print(dataset)
dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=100)


# Define model
model = nn.Linear(11, 1)

# Define a utility function to train the model
def fit(num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):
        for xb,yb in train_dl:
            # Generate predictions
            pred = model(xb)
            loss = loss_fn(pred, yb)
            # Perform gradient descent
            loss.backward()
            opt.step()
            opt.zero_grad()
        print('Training loss: ', loss_fn(model(inputs), targets))

class SimpleNet(nn.Module):
    # Initialize the layers
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(11, 11)
        self.act1 = nn.ReLU() # Activation function
        self.linear2 = nn.Linear(11, 1)
    
    # Perform the computation
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        return x


model = SimpleNet()
opt = torch.optim.SGD(model.parameters(), 1e-5)
loss_fn = F.mse_loss

# fit(1000, model, loss_fn, opt)
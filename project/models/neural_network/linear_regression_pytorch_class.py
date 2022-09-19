import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Convert inputs and targets to tensors
numerical_dataset = pd.read_csv('project/dataframes/numerical_data.csv', index_col=0)
inputs = torch.tensor(numerical_dataset.drop('price_night', axis=1).values).float()  # 890x11
targets = torch.tensor(numerical_dataset['price_night']).float()  # 890x1

# Define dataset
train_ds = TensorDataset(inputs, targets)

# Define data loader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
# print(next(iter(train_dl)))

# Define model
model = nn.Linear(11, 1)
print(model.weight)
print(model.bias)

opt = torch.optim.SGD(model.parameters(), lr=1e-5)  # Define optimizer
loss_fn = F.mse_loss  # Define loss function


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

# Train the model for 100 epochs
fit(100, model, loss_fn, opt)

preds = model(inputs)
print(preds)
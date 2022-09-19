import torch
import pandas as pd
import numpy as np


# Convert inputs and targets to tensors
numerical_dataset = pd.read_csv('project/dataframes/numerical_data.csv', index_col=0)
inputs = torch.tensor(numerical_dataset.drop('price_night', axis=1).values).float()  # 890x11
targets = torch.tensor(numerical_dataset['price_night']).float()  # 890x1

# Weights and biases
W = torch.randn(1, 11, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Define the model
def model(x):
    return x @ W.t() + b

# MSE loss
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

# Train for 100 epochs
for i in range(10000):
    preds = model(inputs)  # Generate predictions
    loss = mse(preds, targets)  # Calculate loss
    print(loss)
    loss.backward()
    # Adjust weights & reset gradients
    with torch.no_grad():
        W -= W.grad * 1e-5
        b -= b.grad * 1e-5
        W.grad.zero_()
        b.grad.zero_()
    
print(preds)





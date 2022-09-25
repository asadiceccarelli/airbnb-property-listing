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

batch_size = len(features)
n_iters = 300
num_epochs = n_iters / (len(features) / batch_size)
num_epochs = int(num_epochs)

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
criterion = nn.MSELoss()

model_learning_rate = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
model_hidden_dims = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

################################
# 5. TRAIN THE MODEL
################################

loss_dict_rate = {}
learning_rate = [1e-4, 1e-5, 1e-6, 1e-7]

for rate in learning_rate:
    model_learning_rate = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    opt = torch.optim.SGD(model_learning_rate.parameters(), lr=rate)
    loss_dict_rate[rate] = []
    for i in range(num_epochs):
        for x_train, y_train in dataloader:
            opt.zero_grad()
            pred = model_learning_rate(x_train)
            loss = criterion(pred, y_train)
            loss.backward()
            opt.step()
        loss_dict_rate[rate].append(criterion(model_learning_rate(features), targets).item())
        print(f'Epoch {i} training loss: {criterion(model_learning_rate(features), targets)}')


hidden_dims = [2, 8, 50, 100]
loss_dict_hidden = {}

for hidden_dim in hidden_dims:
    model_hidden_dims = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    opt = torch.optim.SGD(model_hidden_dims.parameters(), lr=1e-5)
    loss_dict_hidden[hidden_dim] = []
    for i in range(num_epochs):
        for x_train, y_train in dataloader:
            opt.zero_grad()
            pred = model_hidden_dims(x_train)
            loss = criterion(pred, y_train)
            loss.backward()
            opt.step()
        loss_dict_hidden[hidden_dim].append(criterion(model_hidden_dims(features), targets).item())
        print(f'Epoch {i} training loss: {criterion(model_hidden_dims(features), targets)}')

###############################
# 6. PLOTS
###############################

error_line = px.line(
    loss_dict_rate,
    labels={'index': 'Epoch', 'value': 'Mean Squared Error', 'variable': 'Learning Rate'},
    title='Loss plot',
    render_mode='SVG'
    )
error_line.update_layout(template='plotly_dark')
error_line.write_image('README-images/error-line.png', scale=20)

hidden_dim_line = px.line(
    loss_dict_hidden,
    labels={'index': 'Epoch', 'value': 'Mean Squared Error', 'variable': 'Hidden neurons'},
    title='Loss plot',
    render_mode='SVG'
    )
hidden_dim_line.update_layout(xaxis_range=[0,100], template='plotly_dark')
hidden_dim_line.write_image('README-images/hidden-dim-line.png', scale=20)

# Modelling Airbnb's Property Listing Dataset

Creating a multimodal deep learning structure that processes text, images and tabular data to predict the type of property from an Airbnb listing.


## Data preparation

<p align='center'>
  <img src='README-images/airbnb-listing-dashboard.png' width='700'>
</p>

> A Tableau dashboard showing initial insights into the data.

The 'structured' data is downloaded as a `.csv` file from an AWS S3 bucket containing the following information:
- `ID`: Unique identifier for the listing
- `Category`: The category of the listing
- `Title`: The title of the listing
- `Description`: The description of the listing
- `Amenities`: The available amenities of the listing
- `Location`: The location of the listing
- `Guests`: The number of guests that can be accommodated in the listing
- `Beds`: The number of available beds in the listing
- `Bathrooms`: The number of bathrooms in the listing
- `Price_Night`: The price per night of the listing
- `Cleanliness_rate`: The cleanliness rating of the listing
- `Accuracy_rate`: How accurate the description of the listing is, as reported by previous guests
- `Location_rate`: The rating of the location of the listing
- `Check-in_rate`: The rating of check-in process given by the host
- `Value_rate`: The rating of value given by the host
- `Amenities_count`: The number of amenities in the listing
- `URL`: The URL of the listing
- `Bedrooms`: The number of bedrooms in the listing

The file `tabular_data.py` contains the function `clean_data()` which will take a DataFrame as an input, and return a cleaned DataFrame. This is saved in the `dataframes` directory as `cleaned_dataset.csv`. A Tableau dashboard is created to take an overview at the data provided.

The function `create_numerical_dataset()` will drop all columns containing non-numerical data and `load_airbnb()` will return the tuple `(features, labels)` ready to train a regression model in the next section.


## Creating a regression model

The file `regression_modelling.py` will contain all the functions necessary to compare and select the best regression model used to predict the price per night of a listing. First, a linear regression model is trained and the baseline metrics are calculated using `get_baseline_score()`:
- Validation RMSE: 99.583
- Validation R2 score: 0.42181

As there is an element of randomness in the splitting of the dataset and in the models, each model will be tuned multiple times with different random states. Then, an average taken in order to find the optimum hyperparameters that will give the most accurate results across the largest number of different seeds. The function `repeat_tuning()` will take the number of different seeds to tune over before using GridSearchCV with a KFolds cross-validator with 5 splits and saving the best hyperparameters for each model. Next, each model is trained a set number of times using the function `train_model_multiple_times()`. The RMSE and R2 score of each train is saved in a dictionary for each model and stored in `repeated_metrics.json`. Finally, a summary of these metrics is calculated easily by converting the data to a Pandas DataFrame and saving in `summary_metrics.json.` .

The box plot below demonstrates the importance of testing of a range of different seeds. The overlapping ranges shows that if just one seed is used, the rank of each model could in theory be in any order when using RMSE as a comarison.

<p align='center'>
  <img src='README-images/regression-box-plot.png' width='500'>
</p>

> A box plot to show to show the range, median and interquartile range of regression models.

Whilst it is evident that the using Decision Trees has the largest error, there is not a great deal to separate the other three regression models. In this case, Linear Regression is selected as the best model according to Occam's Razor, which states that we should prefer models with fewer coefficients over complex models like ensembles. This is because simpler machine learning models are expected to generalise better, and have less a tendancy to overfit.

<p align='center'>
  <img src='README-images/regression-mean-rmse.png' width='500'>
</p>

> A comparison of the RMSE of each model calculated from the validation set.

Whilst maybe slightly overfit, the Linear Regression model is a good fit for our data as the error in both the training and testing sets are similar. The R2 score is interesting slightly higher in the testing set meaning there is less variance, but again this is only by a small amount.

<p align='center'>
  <img src='README-images/train-test-rmse.png' width='400'>
  <img src='README-images/train-test-r2.png' width='400'>
</p>

> A comparison of the train and test sets' RMSE and R2 scores.


## Creating a classification model

`classification_modelling.py` contains all the functions needed to create and tune classification models used to predict the category (treehouse, chalet, offbeat, beachfront or amazing pools). It is almost identical to the file `regression_modelling.py`, with the regression models replaced with their classification counterpart. For these models, the price per night will also be included as a feature. Using Logistic Regression gives the baseline metrics from the validation set as:
- Validation accuracy score: 0.46296
- Validation precision score: 0.52363
- Validation recall score: 0.40310
- Validation F1 score: 0.38426

Once again, each model is tuned with 10 different seeds so that the models can be more accurately compared.

<p align='center'>
  <img src='README-images/classification-box-plot.png' width='500'>
</p>

> A box plot to show to show the range, median and interquartile range of classification models.

As XGBoost requires the labels to be a numeric value, and therefore a label encoder is used to map the categories to the label space `[0, 1, 2, 3, 4]`. As an alternative, one-hot encoding could be used to avoid the model understanding a ranking between the outputs.

```python
label_encoder = LabelEncoder().fit(y)
label_encoded_y = label_encoder.transform(y)
```

> Encoding the label space using `LabelEncoder`.

<p align='center'>
  <img src='README-images/classification-mean-accuracy-score.png' width='400'>
  <img src='README-images/classification-mean-f1-score.png' width='400'>
</p>

> A comparison of the mean accuracy and F1 score of each model calculated from the validation set.

Once again, the Decision Trees model is by far the worst model, with not much separating the other three in terms of accuracy. Whilst XGBoost does have a higher average F1 score, Logistic Regression will be taken as the best model according to Occam's Razor once more.

<p align='center'>
  <img src='README-images/train-test-accuracy.png' width='400'>
  <img src='README-images/train-test-f1.png' width='400'>
</p>

> A comparison of the train and test sets' accuracy and F1 scores.

As the training set performs better than the testing set, it seems that the model is overfitting. There are several options to combat this:
- Alter the train-validation-test split
- Reduce the number of features
- Incoorporate regularisation (`C` hyperparameter in Logistic Regression)


## Creating an FeedForward Artificial Neural Network

<p align='center'>
  <img src='README-images/FNN.png' width='300'>
</p>

Feedforward neural networks are also known as Multi-layered Network of Neurons (MLN). These network of models are called feedforward because the information only travels forward in the neural network, through the input nodes then through the hidden layers and finally through the output nodes.

The Pytorch NN module will be used to create this model.

### Creating the DataLoader

The class `PriceNightDataset` is created as a subclass `torch.utils.data.Dataset` which covers the data in a tuple and enables the access the index of each sample, as well as the length of the datasets. This class also contains the assertion `len(X) == len(y)` to ensure that the features and targets are of equal length.

```py
dataset = price_night_Dataset(inputs, features)
```

> Creating the dataset from the `price_night_Dataset` class.

In deep learning, batches of data are used (usually as much as can fit onto a GPU). Using `torch.utils.data.DataLoader` as an iterable, the dataset is batched so it is more easily consumed by the neural network. The bath size will initially be set to a value of 100 and to be shuffled before each iteration, but the effect of different sizes and not shuffling will be inspected later in the project. 

```py
dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=100
```

> Creating the dataloader from the dataset.

### Creating the network architecture

The neural network is created as a class `FeedforwardNeuralNetModel` using Object Orientated Programming (OOP). The layers are defined in the `init` function and the forward pass is defined in the `forward` function, which is invoked automatically when the class is called. Using `super(FeedforwardNeuralNetModel, self).__init__`, these functions are possible as the class `nn.Module` from torch is inherited. Two linear hidden layers (`linear1` and `linear2`) are used with a `ReLU` activation function (`act1`).

N.B. `nn.Linear` takes a input shape and output shape and produces a weight and bias term for the specified shape.

### Instantiating the model

The input and output dimensions are determined by the number of features to targets, which in this case are 11 and 1 respectively. Determining the dimension of the hidden layer requires a little more thought. Too few hidden neurons and there is insufficient model capacity to predict competently. However a bigger model does not necessarily always equate to a better model. A bigger model will require more training samples to learn and converge to a good model (also called the curse of dimensionality), hence the optimal number will depend on the problem. There are many genereal 'rule-of thumb' methods such as:
- The number of hidden neurons should be between the size of the input layer and the size of the output layer.
- The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
- The number of hidden neurons should be less than twice the size of the input layer.

A single hidden layer containing 8 neurons will be used initially.

```py
model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
```

> Instantiating the model.

### Setting model parameters

1. `criterion = nn.MSELoss()`
    - As this is a regression problem, the Mean Squared Error (MSE) will be used as a loss function.
2. `optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)`
    - `optim.SGD` is used instead of manually manipulating the weights and biases, and the learning rate is set to 1e-5.
3. `epochs = 1500`
    - The number of iterations for training.

### Training the model

The model is trained by iterating through a `for` loop for each epoch:

1. Set gradient w.r.t. parameters to zero with the `.zero_grad` method
3. Predict an array of targets based on the features
4. Calculate loss
5. Calculate gradients w.r.t. parameters with the `.backward()` method
6. Update parameters using gradients with the `.step()` method

```py
for i in range(epochs):
  for x_train, y_train in dataloader:
    opt.zero_grad()
    pred = model(x_train)
    loss = criterion(pred, y_train)
    loss.backward()
    opt.step()
```

> The training loop.

Each time the model is trained, the RMSE, R2 score, training duration and average inference latency is saved in a `metrics.json` file along with the model in a directory named after the time and date, e.g. a model trained on the 1st of January at 08:00:00 would be saved in a folder called `models/neural_networks/regression/2018-01-01_08:00:00`

The model is saved using `torch.save()` which saves the entire module using Python's `pickle` module. The average inference latency is calculated using the memory efficient incremental average to avoid storing a list of all latency frequencies.

```py
inference_latency_start = time()
pred = model(X)
inference_latency = time() - inference_latency_start
counter += 1
mean_inference_latency += (inference_latency - mean_inference_latency) / counter
```

> Calculating mean inference latency using an incremental average.

### Tuning the parameters

<p align='center'>
  <img src='README-images/error-line.png' width='500'>
  <img src='README-images/hidden-dim-line.png' width='500'>
</p>

> The Mean Square Error as the learning rate and number of hidden neurons is varied.

A smaller learning rate means the model takes a larger number of epoch to converge, however too small a learning rate will lead to unstable oscillations in the MSE. It seems for this model containing 8 neurons in the hidden layer, 1e-5 is the optimal learning rate.

Varying the the dimension of the hidden layer has a less pronounced effect: a larger number of neurons will lead to a faster convergence, however has a far longer computation time. Too few neurons and the model will not converge. As the improvement on final MSE with more a large number of neurons for this probelem is so small, the original number of 8 neurons will be used for the model.

The function `generate_nn_config()` creates a range of different configurations in the form of dictionaries, varying the learning rate and set up of the hidden layers. Then, `find_best_nn()` will iterate through these configurations, creating and training the feed forward neural network each time and save the model, hyperparameters and metrics. Finally it will check through each of these runs and output the hyperparameters of the model with the lowest validation set RMSE to the log. In this case, the best configuration for the hidden layers is in the form [6, 2] with a learning rate of 1e-5.

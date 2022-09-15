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

The file `regression_modelling.py` will contain all the functions necessary to compare and select the best regression model used to predict the price per night of a listing. First, linear regression is used to calculate a baseline scores from the validation set:
- Validation RMSE: 128.30976798890836
- Validation R2 score: 0.3152634300318299

The function `tune_regression_model_hyperparameters()` will take a model, training, validation and testing sets, a dictionary of hyperparameter ranges to be tuned and a location of where to save calculated data as its parameters. Then, using `GridSearchCV` with a `KFolds` cross-validator, it will save the model, best parameters and performance metrics (RMSE and R2 score) to a folder in the `regression_models` directory. `evaluate_all_models()` will go through the models one by one and tune the respective hyperparameters, before `find_best_model()` will search through the `regression_models` directory and return the model with the lowest RMSE.

As can be seen from the chart below, the Random Forest Regressor has the lowest RMSE value out of the four models tested.

<p align='center'>
  <img src='README-images/regression-rmse.png' width='500'>
</p>

> A comparison of the RMSE of each model calculated from the validation set.


## Creating a classification model

`classification_modelling.py` contains all the functions needed to create and tune classification models used to predict the category (treehouse, chalet, offbeat, beachfront or amazing pools). It is almost identical to the file `regression_modelling.py`, with the regression models replaced with their classification counterpart. For these models, the price per night will also be included as a feature. Using logistic regression gives the baseline metrics from the validation set as:
- Validation accuracy score: 0.46296296296296297
- Validation precision score: 0.5236257309941521
- Validation recall score: 0.40310439560439554
- Validation F1 score: 0.3842583070524247

Comparing the four models demonstrates that XGBoost provides the most accurate results. It should be noted that XGBoost requires the labels to be a numeric value, and therefore a label encoder is used to map the categories to the label space `[0, 1, 2, 3, 4]`.

```python
label_encoder = LabelEncoder().fit(y)
label_encoded_y = label_encoder.transform(y)
```

> Encoding the label space using `LabelEncoder`.

<p align='center'>
  <img src='README-images/classification-accuracy.png' width='500'>
</p>

> A comparison of the accuracy score of each model calculated from the validation set.

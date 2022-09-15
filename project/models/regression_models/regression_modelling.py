import logging
import math
import joblib
import json
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)


def load_airbnb(df, labels):
    """Splits the DataFrame into features and labels, ready
        to train a model.
    Args:
        df (DataFrame): Complete DataFrame.
        labels (list): Column names of labels
    Returns:
        (tuple): Tuple containing features and labels.
    """
    logging.info('Splitting into features and labels...')
    labels_df = df[labels]
    features_df = df.drop(labels, axis=1)
    return (features_df, labels_df)


numerical_dataset = pd.read_csv('project/dataframes/numerical_data.csv', index_col=0)
data = load_airbnb(numerical_dataset, 'price_night')
X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.2, random_state=13)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.3, random_state=13)


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
        'Test RMSE:': rmse_test,
        'Test R2 score': r2_test
        }
    return metrics


def save_model(model, hyperparameters, metrics, folder):
    """saves the information of a tuned regression model.
    Args:
        model (class): Saved as a .joblib file.
        hyperparameters (dict): Saved as a .json file.
        metrics (dict): Saved as a .json file.
        folder (str): The directory path of where to save the data.
    """
    logging.info('Saving data...')
    joblib.dump(model, f'{folder}/model.joblib')
    with open(f'{folder}/hyperparameters.json', 'w') as outfile:
        json.dump(hyperparameters, outfile)
    with open(f'{folder}/metrics.json', 'w') as outfile:
        json.dump(metrics, outfile)


def get_baseline_score(regression_model, sets, folder):
    """Tunes the hyperparameters of a regression model and saves the information.
    Args:
        model (class): The regression model to be as a baseline.
        sets (list): List in the form [X_train, y_train, X_validation,
            y_validation, X_test, y_test].
        folder (str): The directory path of where to save the data.
    Returns:
        metrics (dict): Training, validation and testing performance metrics.
    """
    logging.info('Calculating baseline score...')
    model = regression_model().fit(sets[0], sets[1])
    y_train_pred = model.predict(sets[0])
    y_validation_pred = model.predict(sets[2])
    y_test_pred = model.predict(sets[4])

    best_params = model.get_params()
    metrics = calculate_regression_metrics(
        sets[1], y_train_pred,
        sets[3], y_validation_pred,
        sets[5], y_test_pred
    )
    save_model(model, best_params, metrics, folder)
    return metrics


def tune_regression_model_hyperparameters(regression_model, sets, hyperparameters, folder):
    """Tunes the hyperparameters of a regression model and saves the information.
    Args:
        regression_model (class): The regression model to be tuned.
        sets (list): List in the form [X_train, y_train, X_validation,
            y_validation, X_test, y_test].
        hyperparameters (dict): Keys as a list of hyperparameters to be tested.
        folder (str): The directory path of where to save the data.
    Returns:
        best_params (dict): The hyperparameters of the most accurate model
        metrics (dict): Training, validation and testing performance metrics.
    """
    logging.info('Performing GridSearch with KFold...')
    model = regression_model(random_state=13)
    kfold = KFold(n_splits=5, shuffle=True, random_state=13)
    clf = GridSearchCV(model, hyperparameters, cv=kfold)

    best_model = clf.fit(sets[0], sets[1])
    y_train_pred = best_model.predict(sets[0])
    y_validation_pred = best_model.predict(sets[2])
    y_test_pred = best_model.predict(sets[4])

    best_params = best_model.best_params_
    metrics = calculate_regression_metrics(
        sets[1], y_train_pred,
        sets[3], y_validation_pred,
        sets[5], y_test_pred
    )

    save_model(best_model, best_params, metrics, folder)
    return best_params, metrics


def evaluate_all_models():
    """Tunes the hyperparameters of DecisionTreeRegressor, RandomForestRegressor
        and XGBRegressor before saving the best model as a .joblib file, and
        best hyperparameters and performance metrics as .json files.
    """
    logging.info('Evaluating models...')
    tune_regression_model_hyperparameters(
        DecisionTreeRegressor,
        [X_train, y_train, X_validation, y_validation, X_test, y_test],
        dict(max_depth=list(range(1, 10))),
        'project/models/regression_models/decision_tree_regressor')

    tune_regression_model_hyperparameters(
        RandomForestRegressor,
        [X_train, y_train, X_validation, y_validation, X_test, y_test],
        dict(
            n_estimators=list(range(80, 90)),
            max_depth=list(range(1, 10)),
            bootstrap=[True, False],
            max_samples = list(range(40, 50))),
        'project/models/regression_models/random_forest_regressor'
    )

    tune_regression_model_hyperparameters(
        xgb.XGBRegressor,
        [X_train, y_train, X_validation, y_validation, X_test, y_test],
        dict(
            n_estimators=list(range(10, 30)),
            max_depth=list(range(1, 10)),
            min_child_weight=list(range(1, 5)),
            gamma=list(range(1, 3)),
            learning_rate=np.arange(0.1, 0.5, 0.1)),
        'project/models/regression_models/xgboost_regressor'
    )


def find_best_model():
    """Searches through the regression_models directory to find the model
        with the smallest RMSE value for the validation set (best model).
    Returns:
        best_model (class): Loads the model.joblib file.
        best_hyperparameters (dict): Loads the hyperparameters.json file.
        best_metrics (dict): Loads the metrics.json file.
    """
    logging.info('Finding best model...')
    paths = glob.glob('project/models/regression_models/*/metrics.json')
    rmse = {}
    for path in paths:
        model = path[33:-13]
        with open(path) as file:
            metrics = json.load(file)
        rmse[model] = metrics['Validation RMSE']

    best_model_name = min(rmse, key=rmse.get)
    best_model = joblib.load(f'project/models/regression_models/{best_model_name}/model.joblib')
    with open(f'project/models/regression_models/{best_model_name}/hyperparameters.json', 'rb') as file:
            best_hyperparameters = json.load(file)
    with open(f'project/models/regression_models/{best_model_name}/metrics.json', 'rb') as file:
            best_metrics = json.load(file)
    return best_model, best_hyperparameters, best_metrics


def compare_rmse():
    """Plots a bar chart to compare validation RMSE of regression
    models trained.
    """
    logging.info('Plotting graphs...')
    paths = glob.glob('project/models/regression_models/*/metrics.json')
    rmse = {}
    for path in paths:
        model = path[33:-13]
        with open(path) as file:
            metrics = json.load(file)
        rmse[model] = metrics['Validation RMSE']

    fig = px.bar(
        x=rmse.values(),
        y=rmse.keys(),
        labels={'x': 'Validation set RMSE', 'y': 'Regression model', 'color': 'RMSE'},
        title='Comparing the Root Mean Squared Error (RMSE) of different models',
        color=rmse.values(),
        color_continuous_scale='solar_r'
        )
    fig.update_layout(template='plotly_dark', yaxis={'categoryorder': 'total descending'})
    fig.update_xaxes(range=[100, 140])
    fig.write_image('README-images/regression-rmse.png', scale=20)


if __name__ == '__main__':
    evaluate_all_models()
    print(find_best_model())
    compare_rmse()

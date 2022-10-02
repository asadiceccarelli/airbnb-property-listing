import sys
import logging
import math
import joblib
import json
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

sys.path.append('project/data_preparation')
from tabular_data import load_airbnb

logging.basicConfig(level=logging.INFO)


def split_dataset(random_state):
    """Splits up the dataset into training, validation and testing datasets
    in the repective ratio of 60:20:20.
    Returns:
        (tuple): (X_train, y_train, X_validation, y_validation, X_test, y_test)
    """
    numerical_dataset = pd.read_csv('project/dataframes/numerical_data.csv', index_col=0)
    data = load_airbnb(numerical_dataset, 'price_night')
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.2, random_state=random_state)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)
    return (X_train, y_train, X_validation, y_validation, X_test, y_test)


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
        'Test RMSE': rmse_test,
        'Test R2 score': r2_test
        }
    return metrics


def get_baseline_score(datasets):
    """Tunes the hyperparameters of a regression model and saves the information.
    Args:
        datasets (tuple): (X_train, y_train, X_validation, y_validation,
        X_test, y_test).
    """
    logging.info('Calculating baseline score...')
    model = LinearRegression().fit(datasets[0], datasets[1])
    y_train_pred = model.predict(datasets[0])
    y_validation_pred = model.predict(datasets[2])
    y_test_pred = model.predict(datasets[4])
    metrics = calculate_regression_metrics(
        datasets[1], y_train_pred,
        datasets[3], y_validation_pred,
        datasets[5], y_test_pred
    )
    joblib.dump(model, open('project/models/regression_models/linear_regression/model.joblib', 'wb'))
    json.dump(metrics, open('project/models/regression_models/linear_regression/baseline_metrics.json', 'w'))


def tune_regression_model_parameters(regression_model, sets, parameters, seed):
    """Tunes the hyperparameters of a regression model and saves the information.
    Args:
        regression_model (class): The regression model to be tuned.
        sets (tuple): (X_train, y_train, X_validation, y_validation,
            X_test, y_test).
        parameters (dict): Keys as a list of hyperparameters to be tested.
        seed (int): The random state of the regression model.
    Returns:
        best_params (dict): The optimal hyperparameters for this model.
    """
    logging.info(f'Performing GridSearch with KFold for {regression_model}...')
    model = regression_model(random_state=seed)
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    grid_search = GridSearchCV(model, parameters, cv=kfold)
    grid_search.fit(sets[0], sets[1])
    best_params = grid_search.best_params_
    return best_params


def evaluate_all_models(datasets, seed):
    """Tunes the hyperparameters of DecisionTreeRegressor, RandomForestRegressor
        and XGBRegressor before saving the best model as a .joblib file, and
        best hyperparameters and performance metrics as .json files.
    Args:
        datasets (list): (X_train, y_train, X_validation, y_validation,
            X_test, y_test).
        seed (int): The random state of the regression model.
    Returns:
        regression_models (tuple): A tuple of the tuned regression models,
            each one containing (best_model, best_params, metrics).
    """
    logging.info('Evaluating models...')
    decision_tree = tune_regression_model_parameters(
        DecisionTreeRegressor,
        datasets,
        dict(max_depth=list(range(1, 5))),
        seed = seed
    )

    random_forest = tune_regression_model_parameters(
        RandomForestRegressor,
        datasets,
        dict(
            n_estimators=list(range(75, 100)),
            max_depth=list(range(5, 17)),
            max_samples = list(range(45, 55)),
        ),
        seed = seed
    )

    xgboost = tune_regression_model_parameters(
        xgb.XGBRegressor,
        datasets,
        dict(
            n_estimators=list(range(15, 33)),
            max_depth=list(range(1, 7)),
            min_child_weight=list(range(1, 15)),
            learning_rate=np.arange(0.1, 1.1, 0.1),
        ),
        seed = seed
    )

    regression_models = {
        'decision_tree_regressor': decision_tree,
        'random_forest_regressor': random_forest,
        'xgboost_regressor': xgboost
    }
    return regression_models


def repeat_tuning(num_seeds):
    """Tune and train each model multiple times, each time with a
    different seed. Saves the metrics in a dictionary.
    Args:
        num_seeds (int): Number of different seeds to train the model.
    """
    seeds = list(range(num_seeds))
    seeds = [6, 5, 3]
    for seed in seeds:
        logging.info(f'Using seed {seed}:')
        datasets = split_dataset(seed)
        regression_models = evaluate_all_models(datasets, seed)
        for model in regression_models:
            best_params = regression_models[model]
            print(best_params)
            with open(f'project/models/regression_models/{model}/seeds_tested/{seed}', 'w') as outfile:
                json.dump(best_params, outfile)


def get_average_parameters():
    """Searches through the seeds tested and averages out the optimum
    hyperparameters which will be used to train the models. Saves as a
    dictionary in hyperparameters.json.
    """
    models = glob.glob('project/models/regression_models/*')
    models = [i for i in models if i not in (
        'project/models/regression_models/regression_modelling.py',
        'project/models/regression_models/linear_regression',
        'project/models/regression_models/regression_graphs.ipynb'
        )]

    paths = []
    for model in models:
        model_name = model.split('/')[-1]
        paths = glob.glob(f'{model}/seeds_tested/*')
        parameter_list = []
        for path in paths:
                with open(path) as file:
                        parameter_list.append(json.load(file))
        df = pd.DataFrame(parameter_list)
        mean_parameters_dict = df.mean().to_dict()

        path = f'project/models/regression_models/{model_name}/hyperparameters.json'
        json.dump(mean_parameters_dict, (path, 'w'))


def save_best_model(datasets):
    """Saves the best models for each model as a .joblib file"""
    decision_tree = DecisionTreeRegressor(
        max_depth=1,
        random_state=13
    )
    model = decision_tree.fit(datasets[0], datasets[1])
    joblib.dump(model, open('project/models/regression_models/decision_tree_regressor/model.joblib', 'wb'))
    
    random_forest = RandomForestRegressor(
        n_estimators=88,
        max_depth=12,
        max_samples=48,
        random_state=13
    )
    model = random_forest.fit(datasets[0], datasets[1])
    joblib.dump(model, open('project/models/regression_models/random_forest_regressor/model.joblib', 'wb'))

    xgboost = xgb.XGBRegressor(
        n_estimators=23,
        max_depth=2,
        min_child_weight=10,
        learning_rate=0.37
    )
    model = xgboost.fit(datasets[0], datasets[1])
    joblib.dump(model, open('project/models/regression_models/xgboost_regressor/model.joblib', 'wb'))


def train_model_multiple_times(no_trains):
    """Trains a model a certain number of times and saves in a
    metrics.json file.
    """
    model_paths = glob.glob('project/models/regression_models/*/model.joblib')
    for path in model_paths:
        model_name = path.split('/')[-2]
        logging.info(f'Training {model_name} {no_trains} times...')
        loaded_model = joblib.load(open(path, 'rb'))
        metrics_dict = {}
        for i in range(no_trains):
            datasets = split_dataset(random_state=None)
            model = loaded_model.fit(datasets[0], datasets[1])
            y_train_pred = model.predict(datasets[0])
            y_validation_pred = model.predict(datasets[2])
            y_test_pred = model.predict(datasets[4])

            metrics_dict[i] = calculate_regression_metrics(
                datasets[1], y_train_pred,
                datasets[3], y_validation_pred,
                datasets[5], y_test_pred
            )
        repeated_metrics_path = f'project/models/regression_models/{model_name}/repeated_metrics.json'
        json.dump(metrics_dict, open(repeated_metrics_path, 'w'))


def calculate_average_metrics():
    '''Calculates the mean, variance and ranges of the validation set
    in repeated_metrics.json file for each model. Saves as in metrics.json.
    '''
    repeated_metrics_path = glob.glob('project/models/regression_models/*/repeated_metrics.json')
    for path in repeated_metrics_path:
        model_name = path.split('/')[-2]
        logging.info(f'Calcualating summary metrics for {model_name}...')
        repeated_metrics = json.load(open(path, 'r'))
        metrics_df = pd.DataFrame(repeated_metrics).transpose().describe()
        metrics_dict = metrics_df.to_dict()
        metrics_path = f'project/models/regression_models/{model_name}/summary_metrics.json'
        json.dump(metrics_dict, open(metrics_path, 'w'))
        

def get_all_data(num_seeds, no_trains):
    """Runs each function in the correct order as to calculate
    all data required from the regression models.
    Args:
        num_seeds (int): The number of different seeds to try in
            order to get the average best parameters.
        no_trains (int): The number of times to train the each
            best regression model.
    """
    datasets = split_dataset(random_state=13)
    get_baseline_score(datasets)
    repeat_tuning(num_seeds)
    get_average_parameters()
    save_best_model(datasets)  # Needs to be edited to ensure complete pipeline
    train_model_multiple_times(no_trains)
    calculate_average_metrics()


if __name__ == '__main__':
    get_all_data(num_seeds=10, no_trains=100)
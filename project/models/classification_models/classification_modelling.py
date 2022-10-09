from itertools import repeat
import logging
import math
import joblib
import json
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO)


def split_dataset(random_state):
    """Splits up the dataset into training, validation and testing datasets
    in the repective ratio of 60:20:20.
    Returns:
        (tuple): (X_train, y_train, X_validation, y_validation, X_test, y_test)
    """
    X = pd.read_csv('project/dataframes/numerical_data.csv', index_col=0)
    y = pd.read_csv('project/dataframes/cleaned_dataset.csv', index_col=0)['category']
    label_encoded_y = LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, label_encoded_y, test_size=0.2, random_state=random_state)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)
    return (X_train, y_train, X_validation, y_validation, X_test, y_test)


def calculate_classification_metrics(y_train, y_train_pred, y_validation, y_validation_pred, y_test, y_test_pred):
    """Calculates the accuray, precision, recall and F1 scores of a classification model.
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
    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred, average='macro')
    recall_train = recall_score(y_train, y_train_pred, average='macro')
    f1_train = f1_score(y_train, y_train_pred, average='macro')
    accuracy_validation = accuracy_score(y_validation, y_validation_pred)
    precision_validation = precision_score(y_validation, y_validation_pred, average='macro')
    recall_validation = recall_score(y_validation, y_validation_pred, average='macro')
    f1_validation = f1_score(y_validation, y_validation_pred, average='macro')
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred, average='macro')
    recall_test = recall_score(y_test, y_test_pred, average='macro')
    f1_test = f1_score(y_test, y_test_pred, average='macro')
    metrics = {
        'Training accuracy score': accuracy_train,
        'Training precision score': precision_train,
        'Training recall score': recall_train,
        'Training F1 score': f1_train,
        'Validation accuracy score': accuracy_validation,
        'Validation precision score': precision_validation,
        'Validation recall score': recall_validation,
        'Validation F1 score': f1_validation,
        'Test accuracy score:': accuracy_test,
        'Test precision score': precision_test,
        'Test recall score score':recall_test,
        'Test F1 score': f1_test
        }
    return metrics


def get_baseline_score(datasets):
    """Tunes the hyperparameters of a classification model and saves the information.
    Args:
        datasets (tuple): (X_train, y_train, X_validation, y_validation,
            X_test, y_test).
    Returns:
    """
    logging.info('Calculating baseline score...')
    model = LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(datasets[0], datasets[1])
    y_train_pred = model.predict(datasets[0])
    y_validation_pred = model.predict(datasets[2])
    y_test_pred = model.predict(datasets[4])

    metrics = calculate_classification_metrics(
        datasets[1], y_train_pred,
        datasets[3], y_validation_pred,
        datasets[5], y_test_pred
    )
    joblib.dump(model, open('project/models/classification_models/logistic_regression/model.joblib', 'wb'))
    json.dump(metrics, open('project/models/classification_models/logistic_regression/baseline_metrics.json', 'w'))


def tune_classification_model_hyperparameters(classification_model, datasets, parameters, seed):
    """Tunes the hyperparameters of a classification model and saves the information.
    Args:
        classification_model (class): The classification model to be tuned.
        datasets (tuple): (X_train, y_train, X_validation, y_validation,
            X_test, y_test).
        parameters (dict): Keys as hyperparameters to be tested.
        seed (int): The random state of the classification model.
    Returns:
        best_params (dict): The optimal hyperparameters for this model.
    """
    logging.info(f'Performing GridSearch with KFold for {classification_model}...')
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    model = classification_model(random_state=seed)
    grid_search = GridSearchCV(model, parameters, cv=kfold)
    grid_search.fit(datasets[0], datasets[1])
    best_params = grid_search.best_params_
    return best_params


def evaluate_all_models(datasets, seed):
    """Tunes the hyperparameters of a classification model and saves the
    information.
    Args:
        classification_model (class): The classification model to be tuned.
        datasets (tuple): (X_train, y_train, X_validation, y_validation,
            X_test, y_test).
        parameters (dict): Keys as hyperparameters to be tested.
        seed (int): The random state of the classification model.
    Returns:
        best_params (dict): The optimal hyperparameters for this model.
    """
    logging.info('Evaluating models...')
    decision_tree = tune_classification_model_hyperparameters(
        DecisionTreeClassifier,
        datasets,
        dict(max_depth=list(range(1, 10))),
        seed=seed
        )
    random_forest = tune_classification_model_hyperparameters(
        RandomForestClassifier,
        datasets,
        dict(
            n_estimators=list(range(120, 135)),
            max_depth=list(range(4, 20)),
            max_samples = list(range(40, 70))),
        seed=seed
    )
    xgboost = tune_classification_model_hyperparameters(
        xgb.XGBClassifier,
        datasets,
        dict(
            n_estimators=list(range(8, 22)),
            max_depth=list(range(1, 9)),
            min_child_weight=list(range(1, 11)),
            learning_rate=np.arange(0.5, 1.3, 0.1)),
        seed=seed
    )
    classification_models = {
        'decision_tree_classifier': decision_tree,
        'random_forest_classifier': random_forest,
        'xgboost_classifier': xgboost
    }
    return classification_models


def repeat_tuning(num_seeds):
    """Tune and train each model multiple times, each time with a
    different seed. Saves the metrics in a dictionary.
    Args:
        num_seeds (int): Number of different seeds to train the model.
    """
    seeds = list(range(num_seeds))
    for seed in seeds:
        logging.info(f'Using seed {seed}:')
        datasets = split_dataset(random_state=seed)
        classification_models = evaluate_all_models(datasets, seed)
        for model in classification_models:
            best_params = classification_models[model]
            print(best_params)
            with open(f'project/models/classification_models/{model}/seeds_tested/{seed}', 'w') as outfile:
                json.dump(best_params, outfile)


def get_average_parameters():
    """Searches through the seeds tested and averages out the optimum
    hyperparameters which will be used to train the models. Saves as a
    dictionary in hyperparameters.json.
    """
    models = glob.glob('project/models/classification_models/*')
    models = [i for i in models if i not in (
        'project/models/classification_models/classification_modelling.py',
        'project/models/classification_models/logistic_regression',
        'project/models/classification_models/classification_graphs.ipynb'
        )]  # Only list directories containing models
    paths = []
    for model in models:
        model_name = model.split('/')[-1]
        paths = glob.glob(f'{model}/seeds_tested/*')
        parameter_list = []
        for path in paths:
            parameter_list.append(json.load(open(path, 'r')))
        df = pd.DataFrame(parameter_list)
        mean_parameters_dict = df.mean().to_dict()
        with open(f'project/models/classification_models/{model_name}/hyperparameters.json', 'w') as outfile:
            json.dump(mean_parameters_dict, outfile)


def save_best_model(datasets):
    """Saves the best models for each model as a .joblib file"""
    decision_tree = DecisionTreeClassifier(
        max_depth=6,
        random_state=13
    )
    model = decision_tree.fit(datasets[0], datasets[1])
    joblib.dump(model, open('project/models/classification_models/decision_tree_classifier/model.joblib', 'wb'))
    
    random_forest = RandomForestClassifier(
        n_estimators=123,
        max_depth=8,
        max_samples=54,
        random_state=13
    )
    model = random_forest.fit(datasets[0], datasets[1])
    joblib.dump(model, open('project/models/classification_models/random_forest_classifier/model.joblib', 'wb'))

    xgboost = xgb.XGBClassifier(
        n_estimators=14,
        max_depth=3,
        min_child_weight=7,
        learning_rate=0.83
    )
    model = xgboost.fit(datasets[0], datasets[1])
    joblib.dump(model, open('project/models/classification_models/xgboost_classifier/model.joblib', 'wb'))


def train_model_multiple_times(no_trains):
    """Trains a model a certain number of times and saves in a
    metrics.json file.
    """
    model_paths = glob.glob('project/models/classification_models/*/model.joblib')
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

            metrics_dict[i] = calculate_classification_metrics(
                datasets[1], y_train_pred,
                datasets[3], y_validation_pred,
                datasets[5], y_test_pred
            )
        with open(f'project/models/classification_models/{model_name}/repeated_metrics.json', 'w') as outfile:
            json.dump(metrics_dict, outfile)


def calculate_average_metrics():
    '''Calculates the mean, variance and ranges of the validation set
    in repeated_metrics.json file for each model. Saves as in metrics.json.
    '''
    repeated_metrics_path = glob.glob('project/models/classification_models/*/repeated_metrics.json')
    for path in repeated_metrics_path:
        model_name = path.split('/')[-2]
        logging.info(f'Calcualating summary metrics for {model_name}...')
        repeated_metrics = json.load(open(path, 'r'))
        metrics_df = pd.DataFrame(repeated_metrics).transpose().describe()
        metrics_dict = metrics_df.to_dict()
        with open(f'project/models/classification_models/{model_name}/summary_metrics.json', 'w') as outfile:
            json.dump(metrics_dict, outfile)


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
    get_all_data()
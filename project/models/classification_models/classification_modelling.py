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


X = pd.read_csv('project/dataframes/numerical_data.csv', index_col=0)
y = pd.read_csv('project/dataframes/cleaned_dataset.csv', index_col=0)['category']
label_encoder = LabelEncoder().fit(y)
label_encoded_y = label_encoder.transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, label_encoded_y, test_size=0.2, random_state=13)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.3, random_state=13)


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


def save_model(model, hyperparameters, metrics, folder):
    """saves the information of a tuned classification model.
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


def get_baseline_score(classification_model, sets, folder):
    """Tunes the hyperparameters of a classification model and saves the information.
    Args:
        model (class): The classification model to be as a baseline.
        sets (list): List in the form [X_train, y_train, X_validation,
            y_validation, X_test, y_test].
        folder (str): The directory path of where to save the data.
    Returns:
        metrics (dict): Training, validation and testing performance metrics.
    """
    logging.info('Calculating baseline score...')
    model = classification_model(multi_class='multinomial', solver='newton-cg', random_state=13).fit(sets[0], sets[1])
    y_train_pred = model.predict(sets[0])
    y_validation_pred = model.predict(sets[2])
    y_test_pred = model.predict(sets[4])

    best_params = model.get_params()
    metrics = calculate_classification_metrics(
        sets[1], y_train_pred,
        sets[3], y_validation_pred,
        sets[5], y_test_pred
    )
    save_model(model, best_params, metrics, folder)
    return metrics


def tune_classification_model_hyperparameters(classification_model, sets, hyperparameters, folder):
    """Tunes the hyperparameters of a classification model and saves the information.
    Args:
        classification_model (class): The classification model to be tuned.
        sets (list): List in the form [X_train, y_train, X_validation,
            y_validation, X_test, y_test].
        hyperparameters (dict): Keys as a list of hyperparameters to be tested.
        folder (str): The directory path of where to save the data.
    Returns:
        best_params (dict): The hyperparameters of the most accurate model
        metrics (dict): Training, validation and testing performance metrics.
    """
    model = classification_model(random_state=13)

    logging.info('Performing GridSearch with KFold...')
    kfold = KFold(n_splits=5, shuffle=True, random_state=13)
    clf = GridSearchCV(model, hyperparameters, cv=kfold)

    best_model = clf.fit(sets[0], sets[1])
    y_train_pred = best_model.predict(sets[0])
    y_validation_pred = best_model.predict(sets[2])
    y_test_pred = best_model.predict(sets[4])

    best_params = best_model.best_params_
    metrics = calculate_classification_metrics(
        sets[1], y_train_pred,
        sets[3], y_validation_pred,
        sets[5], y_test_pred
    )

    save_model(best_model, best_params, metrics, folder)
    return best_params, metrics


def evaluate_all_models():
    """Tunes the hyperparameters of DecisionTreeClassifier, RandomForestClassifier
        and XGBClassifier before saving the best model as a .joblib file, and
        best hyperparameters and performance metrics as .json files.
    """
    
    tune_classification_model_hyperparameters(
        DecisionTreeClassifier,
        [X_train, y_train, X_validation, y_validation, X_test, y_test],
        dict(max_depth=list(range(1, 10))),
        'project/models/classification_models/decision_tree_classifier'
        )

    tune_classification_model_hyperparameters(
        RandomForestClassifier,
        [X_train, y_train, X_validation, y_validation, X_test, y_test],
        dict(
            n_estimators=list(range(120, 130)),
            max_depth=list(range(3, 8)),
            bootstrap=[True, False],
            max_samples = list(range(25, 35))),
        'project/models/classification_models/random_forest_classifier'
    )

    tune_classification_model_hyperparameters(
        xgb.XGBClassifier,
        [X_train, y_train, X_validation, y_validation, X_test, y_test],
        dict(
            n_estimators=list(range(10, 20)),
            max_depth=list(range(1, 5)),
            min_child_weight=list(range(1, 3)),
            gamma=list(range(1, 3)),
            learning_rate=np.arange(0.5, 1.1, 0.1)),
        'project/models/classification_models/xgboost_classifier'
    )


def find_best_model():
    """Searches through the classification_models directory to find the model
        with the highest accuracy value for the validation set (best model).
    Returns:
        best_model (class): Loads the model.joblib file.
        best_hyperparameters (dict): Loads the hyperparameters.json file.
        best_metrics (dict): Loads the metrics.json file.
    """
    logging.info('Finding best model...')
    paths = glob.glob('project/models/classification_models/*/metrics.json')
    accuracy = {}
    for path in paths:
        model = path[37:-13]
        with open(path) as file:
            metrics = json.load(file)
        accuracy[model] = metrics['Validation accuracy score']

    best_model_name = min(accuracy, key=accuracy.get)
    best_model = joblib.load(f'project/models/classification_models/{best_model_name}/model.joblib')
    with open(f'project/models/classification_models/{best_model_name}/hyperparameters.json', 'rb') as file:
            best_hyperparameters = json.load(file)
    with open(f'project/models/classification_models/{best_model_name}/metrics.json', 'rb') as file:
            best_metrics = json.load(file)
    return best_model, best_hyperparameters, best_metrics


def compare_accuracy():
    """Plots a bar chart to compare validation accuracy of classification
    models trained.
    """
    logging.info('Plotting graphs...')
    paths = glob.glob('project/models/classification_models/*/metrics.json')
    accuracy = {}
    for path in paths:
        model = path[37:-13]
        with open(path) as file:
            metrics = json.load(file)
        accuracy[model] = metrics['Validation accuracy score']

    fig = px.bar(
        x=accuracy.values(),
        y=accuracy.keys(),
        labels={'x': 'Validation set accuracy score', 'y': 'Classification model', 'color': 'Accuracy'},
        title='Comparing the accuracy score of different models',
        color=accuracy.values(),
        color_continuous_scale='solar'
        )
    fig.update_layout(template='plotly_dark', yaxis={'categoryorder': 'total descending'})
    fig.update_xaxes(range=[0.2, 0.5])
    fig.write_image('README-images/classification-accuracy.png', scale=20)


if __name__ == '__main__':
    evaluate_all_models()
    print(find_best_model())
    compare_accuracy()
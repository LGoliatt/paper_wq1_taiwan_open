# -*- coding: utf-8 -*-
"""
Machine Learning Experiment for Water Quality Analysis

This script automates the process of training, optimizing, and evaluating
various regression models on the Taiwan Water Quality dataset. It uses Optuna
for hyperparameter tuning and runs experiments multiple times with different
seeds to ensure robust results.

Key Features:
- Fetches and preprocesses data directly from a public Google Sheet.
- Fixes the train/test split for consistent evaluation across runs.
- Utilizes Optuna for efficient hyperparameter optimization.
- Supports multiple models: NGBoost, CatBoost, SVM, ElasticNet, XGBoost.
- Saves detailed results, including metrics, best parameters, and predictions,
  in a JSON file for each run.
- Generates a scatter plot of actual vs. predicted values.
- The entire experiment is repeated 30 times with varying seeds for statistical validity.
"""

import os
import argparse
import json
import logging
import time
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import requests
import seaborn as sns
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from ngboost import NGBRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


def load_and_preprocess_data(test_size: float = 0.25, seed: int = 42):
    """
    Loads and preprocesses the Taiwan Water Quality dataset from a Google Sheet.

    Args:
        test_size (float): The proportion of the dataset to allocate to the test split.
        seed (int): The random state seed for reproducibility of the train/test split.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test.
    """
    key = '1a5DReajqstsnUSUdTcRm8pZqeIP9ZmOct834UcOLmjg'
    link = f'https://docs.google.com/spreadsheet/ccc?key={key}&output=csv'
    r = requests.get(link)
    data = r.content
    df = pd.read_csv(BytesIO(data), header=0)

    # Pivot the table to get features as columns
    cols = ['siteid', 'sampledate', 'itemengabbreviation', 'itemvalue']
    data = df[cols]
    data = data.pivot(index=['siteid', 'sampledate'], columns='itemengabbreviation', values='itemvalue')
    data['site'] = [data.index[i][0] for i in range(len(data))]
    data = data[data['site'] < 1008]

    # Select features and target
    cols = ['EC', 'RPI', 'SS', 'WT', 'pH']
    X_full = data[cols]

    for c in cols:
        X_full[c] = pd.to_numeric(X_full[c], errors='coerce')

    X_full.dropna(inplace=True)

    variable_names = ['EC', 'SS', 'WT', 'pH']
    target_name = 'RPI'

    X = X_full[variable_names]
    y = X_full[target_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    return X_train.values, X_test.values, y_train.values, y_test.values


def objective(trial: optuna.Trial, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> float:
    """
    Objective function for Optuna hyperparameter optimization.

    This function defines the hyperparameter search space for each model and
    evaluates a given set of hyperparameters using 5-fold cross-validation.

    Args:
        trial (optuna.Trial): An Optuna trial object.
        model_name (str): The name of the model to optimize.
        X_train (np.ndarray): The training feature data.
        y_train (np.ndarray): The training target data.

    Returns:
        float: The mean negative MSE from cross-validation, which Optuna will minimize.
    """
    # Use a fixed seed for KFold to ensure fair comparison between trials
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    if model_name == 'NGBoost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'minibatch_frac': trial.suggest_float('minibatch_frac', 0.5, 1.0),
            'natural_gradient': trial.suggest_categorical('natural_gradient', [True, False]),
        }
        model = NGBRegressor(**params, random_state=trial.number)
    elif model_name == 'CatBoost':
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        }
        model = CatBoostRegressor(**params, verbose=0, random_state=trial.number)
    elif model_name == 'SVM':
        params = {
            'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
            'epsilon': trial.suggest_float('epsilon', 0.001, 0.1),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
        }
        model = SVR(**params)
    elif model_name == 'ElasticNet':
        params = {
            'alpha': trial.suggest_float('alpha', 1e-4, 1.0, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 1.0),
        }
        model = ElasticNet(**params, random_state=trial.number)
    elif model_name == 'XGBoost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 1.0),
        }
        model = XGBRegressor(**params, random_state=trial.number)
    else:
        raise ValueError("Unsupported model name")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    # Calculate cross-validation scores
    scores = cross_val_score(pipeline, X_train, y_train, cv=kf,
                             scoring='neg_mean_squared_error', n_jobs=-1)

    return -np.mean(scores)  # Optuna minimizes, so we minimize negative MSE


def optimize_hyperparameters(model_name: str, X_train: np.ndarray, y_train: np.ndarray, n_trials: int = 100, timeout: int = 3600) -> dict:
    """
    Runs the hyperparameter optimization process using Optuna.

    Args:
        model_name (str): The name of the model.
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training target data.
        n_trials (int): The number of optimization trials to run.
        timeout (int): The maximum time in seconds for the optimization.

    Returns:
        dict: A dictionary containing the best hyperparameters found.
    """
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(n_startup_trials=10, seed=42),
        pruner=optuna.pruners.HyperbandPruner()
    )
    study.optimize(
        lambda trial: objective(trial, model_name, X_train, y_train),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=-1  # Use all available CPU cores
    )
    return study.best_params


def train_and_evaluate(model_name: str, best_params: dict, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, seed: int) -> tuple:
    """
    Trains the model with the best hyperparameters and evaluates it on the test set.

    Args:
        model_name (str): The name of the model.
        best_params (dict): The best hyperparameters found by Optuna.
        X_train, y_train: The training data.
        X_test, y_test: The test data.
        seed (int): The random seed for model initialization.

    Returns:
        tuple: A tuple containing the trained pipeline, predictions, MSE, and R-squared score.
    """
    if model_name == 'NGBoost':
        model = NGBRegressor(**best_params, random_state=seed)
    elif model_name == 'CatBoost':
        model = CatBoostRegressor(**best_params, verbose=0, random_state=seed)
    elif model_name == 'SVM':
        model = SVR(**best_params)
    elif model_name == 'ElasticNet':
        model = ElasticNet(**best_params, random_state=seed)
    elif model_name == 'XGBoost':
        model = XGBRegressor(**best_params, random_state=seed)
    else:
        raise ValueError("Unsupported model name")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{model_name} with seed {seed} - MSE: {mse:.4f} - R²: {r2:.4f}")
    return pipeline, y_pred, mse, r2


def plot_results(y_test: np.ndarray, y_pred: np.ndarray, model_name: str, mse: float, r2: float, seed: int):
    """
    Generates and saves a scatter plot of actual vs. predicted values.

    Args:
        y_test (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.
        model_name (str): Name of the model.
        mse (float): Mean Squared Error score.
        r2 (float): R-squared score.
        seed (int): The seed used for the run, for unique filenames.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.title(f"{model_name} (Seed: {seed})\nMSE: {mse:.4f} - R²: {r2:.4f}")
    plt.tight_layout()

    # Save plots
    output_dir = "results"
    plt.savefig(f'{output_dir}/plot_{model_name}_{seed}.png', dpi=300)
    plt.savefig(f'{output_dir}/plot_{model_name}_{seed}.pdf')
    plt.close()


def main(model_name: str, n_trials: int, timeout: int, sample_frac: float, seed: int):
    """
    Main function to run a single experiment.
    """
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Configure logging
    log_file = f"{output_dir}/experiment_{model_name}_{seed}.log"
    logging.basicConfig(
        level=logging.INFO,
        filename=log_file,
        filemode="w",  # Overwrite log for each new run
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info(f"Starting experiment for {model_name} with seed {seed}")

    # Load data with a fixed train/test split seed
    X_train, X_test, y_train, y_test = load_and_preprocess_data(seed=42)

    # Subsample training data if specified
    if sample_frac < 1.0:
        np.random.seed(seed)  # Use the run's seed for subsampling
        idx = np.random.choice(len(X_train), int(len(X_train) * sample_frac), replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
        logging.info(f"Using {sample_frac*100:.1f}% of training data ({len(X_train)} samples).")

    start_time = time.time()

    # Optimize hyperparameters
    best_params = optimize_hyperparameters(model_name, X_train, y_train,
                                           n_trials=n_trials, timeout=timeout)
    logging.info(f"Best parameters for {model_name}: {best_params}")

    # Train and evaluate the final model
    model_pipeline, y_pred, mse, r2 = train_and_evaluate(model_name, best_params, X_train, y_train, X_test, y_test, seed)
    end_time = time.time()
    elapsed_time = end_time - start_time

    logging.info(f"{model_name} - MSE: {mse:.4f} - R²: {r2:.4f}, Total time: {elapsed_time:.2f} s")
    print(f"Total execution time for {model_name} (Seed {seed}): {elapsed_time:.2f} s")

    # Save results to a JSON file
    results = {
        "model": model_name,
        "seed": seed,
        "mse": mse,
        "r2": r2,
        "execution_time_seconds": elapsed_time,
        "best_params": best_params,
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist()
    }
    with open(f"{output_dir}/results_{model_name}_{seed}.json", "w") as f:
        json.dump(results, f, indent=4)

    # Generate and save result plots
    plot_results(y_test, y_pred, model_name, mse, r2, seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run ML experiments for water quality analysis.")
    parser.add_argument("--model", type=str, default='ElasticNet',
                        choices=['NGBoost', 'CatBoost', 'SVM', 'ElasticNet', 'XGBoost'],
                        help="The name of the model to run.")
    parser.add_argument("--n_trials", type=int, default=100,
                        help="Number of optimization trials for Optuna.")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Maximum time in seconds for optimization.")
    parser.add_argument("--sample_frac", type=float, default=1.0,
                        help="Fraction of the training data to use (0.0 to 1.0).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Initial seed for reproducibility.")
    args = parser.parse_args()

    # Run the experiment 30 times with different seeds for robustness
    print(f"--- Starting 30 runs for model: {args.model} ---")
    for i in range(30):
        current_seed = args.seed + i
        main(args.model, args.n_trials, args.timeout, args.sample_frac, current_seed)
    print("--- All 30 runs completed. ---")

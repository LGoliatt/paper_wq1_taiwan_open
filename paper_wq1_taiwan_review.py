''' ENHANCED WATER QUALITY PREDICTION FRAMEWORK Comprehensive machine learning framework with XAI, robust validation, and advanced models '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging
import argparse
import os
from io import BytesIO
import requests
import json
import warnings
warnings.filterwarnings('ignore')
# Fix for numpy compatibility
try:
    np.int = int
    np.float = float
    np.bool = bool
except AttributeError:
    pass
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate, RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from ngboost import NGBRegressor
import optuna
# SHAP with proper error handling
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError as e:
    print(f"SHAP not available: {e}")
    SHAP_AVAILABLE = False
except AttributeError as e:
    print(f"SHAP compatibility issue: {e}")
    SHAP_AVAILABLE = False
from scipy import stats
from scipy.stats import shapiro, jarque_bera
from pandas.plotting import autocorrelation_plot
# Deep Learning imports
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force CPU to avoid GPU issues
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    import tensorflow.keras.backend as K
    from sklearn.utils import resample # For Bagging
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. DeepEnsemble model will be disabled.")

class CNN1D(keras.Model):
    """
    FAST 1D Convolutional Neural Network for tabular data
    ✅ Works with small datasets
    ✅ No sklearn parallelization issues (manual CV)
    ✅ Lightning fast training
    ✅ Perfect for water quality features
    """
   
    def __init__(self, filters=32, kernel_size=2, dense_units=32, lr=0.01, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        self.conv1 = layers.Conv1D(filters, kernel_size, activation='relu', padding='same')
        self.conv2 = layers.Conv1D(filters//2, kernel_size, activation='relu', padding='same')
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dropout = layers.Dropout(0.2)
        self.dense = layers.Dense(dense_units, activation='relu')
        self.final_output = layers.Dense(1)  # Renamed from 'output' to avoid conflict
       
    def call(self, x, training=False):
        x = tf.expand_dims(x, -1) # (batch, features, 1) for Conv1D
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        x = self.dense(x)
        return self.final_output(x)  # Updated to use renamed attribute
   
    def compile_model(self):
        self.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss='mse',
            metrics=['mae']
        )

class CNNRegressor:
    """CNN wrapper compatible with sklearn pipelines."""
   
    def __init__(self, filters=32, kernel_size=2, dense_units=32, lr=0.01, epochs=50):
        self.filters = filters
        self.kernel_size = kernel_size
        self.dense_units = dense_units
        self.lr = np.clip(lr, 1e-3, 0.1)
        self.epochs = min(epochs, 100)
        self.model = None
        self.scaler = StandardScaler()
   
    def fit(self, X, y, **kwargs):
        X_scaled = self.scaler.fit_transform(X)
       
        self.model = CNN1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            dense_units=self.dense_units,
            lr=self.lr
        )
        self.model.compile_model()
       
        # Fast training with early stopping
        self.model.fit(
            X_scaled, y,
            epochs=self.epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=3)
            ],
            verbose=0
        )
        return self
   
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0).flatten()

class DeepEnsembleRegressor:
    """ Robust Deep Ensemble Framework for Regression and Uncertainty Quantification. This model captures both aleatoric and epistemic uncertainty. - Aleatoric Uncertainty: Captured by predicting the mean and variance of a Gaussian distribution for each data point. - Epistemic Uncertainty: Captured by training multiple models (the ensemble) and measuring their disagreement. Uses Bagging (Bootstrap Aggregating) to improve ensemble diversity. """
    def __init__(self, n_models=10, hidden_layers=[64, 64], dropout_rate=0.1, activation='relu', optimizer_cls=keras.optimizers.Adam, learning_rate=1e-3, epochs=200, batch_size=64, early_stopping_patience=15, use_bagging=True, random_state=42):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras is required for DeepEnsembleRegressor")
        self.n_models = n_models
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.optimizer_cls = optimizer_cls
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.use_bagging = use_bagging
        self.random_state = random_state
        self.models = []
        # Define seeds for ensemble reproducibility
        np.random.seed(int(self.random_state))
        seeds_array = np.random.randint(0, 10000, size=self.n_models)
        self.model_seeds = [int(seed) for seed in seeds_array]
    
    def _gaussian_nll_loss(self, y_true, y_pred):
        """ Gaussian Negative Log-Likelihood (NLL) Loss Function. Assume that y_pred consists of [mean, log_variance]. """
        # Separate predicted mean and log(variance)
        mu = y_pred[:, 0:1]
        log_var = y_pred[:, 1:2]
        # Ensure log_var is not too small for stability
        log_var = K.clip(log_var, K.epsilon() - 7, 20)
        # The loss is 0.5 * (log(sigma^2) + (y - mu)^2 / sigma^2)
        loss = 0.5 * (log_var + K.square(y_true - mu) / K.exp(log_var))
        return K.mean(loss)
    
    def _build_model(self, input_dim):
        """ Builds a single neural network model that outputs mean and log(variance). """
        inputs = layers.Input(shape=(input_dim,))
        x = inputs
        for units in self.hidden_layers:
            x = layers.Dense(units, activation=self.activation)(x)
            if self.dropout_rate > 0:
                x = layers.Dropout(self.dropout_rate)(x)
        # Output layer: 2 neurons # 1. Mean (mu) - linear activation (default) # 2. Log Variance (log_var) - linear activation (default) # Predicting log_var is numerically more stable than var.
        outputs = layers.Dense(2, activation='linear')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        optimizer = self.optimizer_cls(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=self._gaussian_nll_loss, metrics=['mae'])
        return model
    
    def fit(self, X, y, validation_split=0.2, verbose=0):
        """ Trains the ensemble of models. """
        self.models = []
        if y.ndim == 1:
            y = y.reshape(-1, 1) # Ensure y has shape (n_samples, 1)
        for i in range(self.n_models):
            seed_value = int(self.model_seeds[i])
            tf.keras.utils.set_random_seed(seed_value)
            if self.use_bagging:
                X_boot, y_boot = resample(X, y, random_state=int(self.model_seeds[i]))
            else:
                X_boot, y_boot = X, y
            model = self._build_model(X.shape[1])
            # Callbacks for robust training
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.early_stopping_patience,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=int(self.early_stopping_patience / 2),
                    min_lr=1e-7
                )
            ]
            if verbose > 0:
                print(f"--- Training Model {i+1}/{self.n_models} ---")
            history = model.fit(
                X_boot, y_boot,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=verbose
            )
            self.models.append(model)
        return self
    
    def predict(self, X):
        """ Makes predictions and decomposes uncertainty. Returns: - mean_pred (np.array): The final predictive mean. - total_std (np.array): The total standard deviation (total uncertainty). - epistemic_std (np.array): The epistemic standard deviation. - aleatoric_std (np.array): The aleatoric standard deviation. """
        if not self.models:
            raise ValueError("The model must be trained (fit) before prediction.")
        # Collect predictions [mu, log_var] from all models # Shape: (n_models, n_samples, 2)
        preds_list = np.array([model.predict(X) for model in self.models])
        # Separate means and log_variances # Shape of both: (n_models, n_samples)
        mus = preds_list[:, :, 0]
        log_vars = preds_list[:, :, 1]
        # Convert log_var to var # Shape: (n_models, n_samples)
        variances = np.exp(log_vars)
        # 1. Final Predictive Mean # Average of the predicted means # Shape: (n_samples,)
        mean_pred = np.mean(mus, axis=0)
        # 2. Aleatoric Uncertainty # Average of the predicted variances # Shape: (n_samples,)
        aleatoric_variance = np.mean(variances, axis=0)
        # 3. Epistemic Uncertainty # Variance (disagreement) among the predicted means # Shape: (n_samples,)
        epistemic_variance = np.var(mus, axis=0)
        # 4. Total Predictive Variance # By the law of total variance: Total = Aleatoric + Epistemic
        total_variance = aleatoric_variance + epistemic_variance
        # Return standard deviations (square root of variance)
        total_std = np.sqrt(total_variance)
        epistemic_std = np.sqrt(epistemic_variance)
        aleatoric_std = np.sqrt(aleatoric_variance)
        return mean_pred
    
    def predict_with_uncertainty(self, X):
        if not self.models:
            raise ValueError("The model must be trained (fit) before prediction.")
        preds_list = np.array([model.predict(X) for model in self.models])
        mus = preds_list[:, :, 0]
        log_vars = preds_list[:, :, 1]
        variances = np.exp(log_vars)
        mean_pred = np.mean(mus, axis=0)
        aleatoric_variance = np.mean(variances, axis=0)
        epistemic_variance = np.var(mus, axis=0)
        total_variance = aleatoric_variance + epistemic_variance
        total_std = np.sqrt(total_variance)
        epistemic_std = np.sqrt(epistemic_variance)
        aleatoric_std = np.sqrt(aleatoric_variance)
        return mean_pred, total_std, epistemic_std, aleatoric_std

def read_wq_taiwan(test_size=0.25, seed=42):
    """ Reading and preprocessing of water quality data. """
    key = '1a5DReajqstsnUSUdTcRm8pZqeIP9ZmOct834UcOLmjg'
    link = 'https://docs.google.com/spreadsheet/ccc?key=' + key + '&output=csv'
    r = requests.get(link)
    data = r.content
    df = pd.read_csv(BytesIO(data), header=0)
    cols = ['siteid', 'sampledate', 'itemengabbreviation', 'itemvalue']
    data = df[cols]
    data = data.pivot(index=['siteid', 'sampledate'], columns='itemengabbreviation', values='itemvalue')
    data['site'] = [data.index[i][0] for i in range(len(data))]
    data = data[data['site'] < 1008]
    cols = ['EC', 'RPI', 'SS', 'WT', 'pH']
    X = data[cols]
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    X.dropna(inplace=True)
    variable_names = ['EC', 'SS', 'WT', 'pH']
    target_names = ['RPI']
    X_train, X_test, y_train, y_test = train_test_split(X[variable_names], X[target_names], test_size=test_size, random_state=seed)
    dataset = {
        'task': 'regression',
        'name': 'WQ Taiwan',
        'feature_names': np.array(variable_names),
        'target_names': target_names,
        'X_train': X_train.values,
        'y_train': y_train.values.ravel(),
        'X_test': X_test.values,
        'y_test': y_test.values.ravel(),
    }
    return dataset['X_train'], dataset['X_test'], dataset['y_train'], dataset['y_test']

def permutation_feature_importance(model_name, model, X_test, y_test, scoring_func=mean_squared_error, n_repeats=100):
    """
    Calculate permutation feature importance for a given model.
   
    Parameters:
        model: Trained model with a `predict` method.
        X_test: Test features (numpy array or pandas DataFrame).
        y_test: True target values (numpy array or pandas Series).
        scoring_func: Scoring function (default is mean squared error).
        n_repeats: Number of times to shuffle each feature (default is 10).
   
    Returns:
        Dictionary containing feature names and their importance scores.
    """
    # Initialize results
    if model_name=='AutoGluon':
        #train_data = pd.DataFrame(X_train, columns=feature_names)
        #train_data[target] = y_train
        test_data = pd.DataFrame(X_test, columns=feature_names)
        test_data[target] = y_test
        baseline_score = scoring_func(y_test, model.predict(test_data).values)
    elif model_name=='H2O':
        test_data = pd.DataFrame(X_test, columns=feature_names)
        test_data[target] = y_test
        test_df = h2o.H2OFrame(test_data)
        baseline_score = scoring_func(y_test, model.predict(test_df).as_data_frame().values.ravel())
    else:
        baseline_score = scoring_func(y_test, model.predict(X_test))
    feature_importances = {feature: [] for feature in range(X_test.shape[1])}
   
    for feature_idx in range(X_test.shape[1]):
        for _ in range(n_repeats):
            # Shuffle the column
            X_test_shuffled = X_test.copy()
            np.random.shuffle(X_test_shuffled[:, feature_idx])
           
            # Predict and score with shuffled feature
            if model_name=='AutoGluon':
                test_data = pd.DataFrame(X_test_shuffled, columns=feature_names)
                test_data[target] = y_test
                shuffled_score = scoring_func(y_test, model.predict(test_data).values)
            elif model_name=='H2O':
                test_data = pd.DataFrame(X_test_shuffled, columns=feature_names)
                test_data[target] = y_test
                test_df = h2o.H2OFrame(test_data)
                shuffled_score = scoring_func(y_test, model.predict(test_df).as_data_frame().values.ravel())
            else:
                shuffled_score = scoring_func(y_test, model.predict(X_test_shuffled))
           
            # Store the difference from baseline
            feature_importances[feature_idx].append(shuffled_score - baseline_score)
   
    # Compute mean importance for each feature
    feature_importances_mean = {
        feature: np.mean(importances) for feature, importances in feature_importances.items()
    }
   
    return feature_importances_mean


def plot_feature_analysis(results, feature_names):
    """ Create comprehensive feature analysis plots """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    # Correlation plot
    correlations = results['correlation_with_target']
    sns.barplot(x=correlations.values, y=correlations.index, ax=axes[0,0])
    axes[0,0].set_title('Feature Correlation with Target')
    axes[0,0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    # Feature importance
    importance = results['rf_importance']
    sns.barplot(x=list(importance.values()), y=list(importance.keys()), ax=axes[0,1])
    axes[0,1].set_title('Random Forest Feature Importance')
    # PCA variance
    axes[1,0].plot(range(1, len(results['pca_variance_ratio'])+1), np.cumsum(results['pca_variance_ratio']), 'bo-')
    axes[1,0].set_title('PCA Cumulative Variance')
    axes[1,0].set_xlabel('Number of Components')
    axes[1,0].set_ylabel('Cumulative Variance')
    axes[1,0].grid(True, alpha=0.3)
    # Feature scores comparison
    features = list(feature_names)
    corr_values = [results['correlation_with_target'][f] for f in features]
    imp_values = [results['rf_importance'][f] for f in features]
    x = np.arange(len(features))
    width = 0.35
    axes[1,1].bar(x - width/2, corr_values, width, label='Correlation', alpha=0.7)
    axes[1,1].bar(x + width/2, imp_values, width, label='RF Importance', alpha=0.7)
    axes[1,1].set_xlabel('Features')
    axes[1,1].set_ylabel('Scores')
    axes[1,1].set_title('Feature Importance vs Correlation')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(features, rotation=45)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def get_model_configuration():
    """ Returns comprehensive model configuration with theoretical justification """
    models = {
        'LinearRegression': {
            'model': LinearRegression(),
            'justification': 'Baseline linear model for establishing performance benchmarks',
            'complexity': 'low',
            'needs_scaling': True
        },
        'ElasticNet': {
            'model': ElasticNet(),
            'justification': 'Linear model with L1 and L2 regularization for handling multicollinearity',
            'complexity': 'low',
            'needs_scaling': True
        },
        'SVM': {
            'model': SVR(),
            'justification': 'Effective for high-dimensional spaces and non-linear relationships using kernel trick',
            'complexity': 'medium',
            'needs_scaling': True
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42),
            'justification': 'Ensemble of decision trees, robust to outliers and captures non-linear relationships',
            'complexity': 'medium',
            'needs_scaling': False
        },
        'XGBoost': {
            'model': XGBRegressor(random_state=42),
            'justification': 'Gradient boosting with regularization, handles missing values, state-of-art for tabular data',
            'complexity': 'high',
            'needs_scaling': False
        },
        'NGBoost': {
            'model': NGBRegressor(),
            'justification': 'Probabilistic forecasting with natural gradient boosting, provides uncertainty quantification',
            'complexity': 'high',
            'needs_scaling': True
        },
        'LightGBM': {
            'model': LGBMRegressor(random_state=42),
            'justification': 'Gradient boosting framework optimized for efficiency and large datasets',
            'complexity': 'medium',
            'needs_scaling': False
        },
        'CatBoost': {
            'model': CatBoostRegressor(random_state=42, verbose=0),
            'justification': 'Handles categorical features naturally, robust to hyperparameter tuning',
            'complexity': 'medium',
            'needs_scaling': False
        },
        'CNN': {
            'model': CNNRegressor(filters=32, lr=0.01),
            'justification': 'Fast 1D Convolutional Neural Network - perfect for tabular features with local patterns',
            'complexity': 'high',
            'needs_scaling': False # Handled internally
        },
    }
    # Add DeepEnsemble if TensorFlow is available
    if TENSORFLOW_AVAILABLE:
        models['DeepEnsemble'] = {
            'model': DeepEnsembleRegressor(),
            'justification': 'Multiple neural networks for robust predictions and uncertainty estimation',
            'complexity': 'high',
            'needs_scaling': True
        }
    return models

def objective(trial, model_name, X_train, y_train):
    """ Objective function for hyperparameter optimization with Optuna. Performs 5-fold cross-validation and minimizes MSE. """
    try:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        models_config = get_model_configuration()
        if model_name == 'NGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'minibatch_frac': trial.suggest_float('minibatch_frac', 0.1, 1.0),
                'natural_gradient': trial.suggest_categorical('natural_gradient', [True, False]),
            }
            model = NGBRegressor(**params)
        elif model_name == 'CatBoost':
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            }
            model = CatBoostRegressor(**params, verbose=0, early_stopping_rounds=20)
        elif model_name == 'SVM':
            params = {
                'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
                'epsilon': trial.suggest_float('epsilon', 0.001, 0.1),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
            }
            model = SVR(**params)
        elif model_name == 'ElasticNet':
            params = {
                'alpha': trial.suggest_float('alpha', 1e-4, 1e0, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 1.0),
            }
            model = ElasticNet(**params, random_state=42)
        elif model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 1.0),
            }
            model = XGBRegressor(**params, random_state=42)
        elif model_name == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            }
            model = RandomForestRegressor(**params, random_state=42)
        elif model_name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            }
            model = LGBMRegressor(**params, random_state=42)
        elif model_name == 'DeepEnsemble' and TENSORFLOW_AVAILABLE:
            params = {
                'n_models': trial.suggest_int('n_models', 3, 8),
                'hidden_layers_1': trial.suggest_int('hidden_layers_1', 4, 64),
                'hidden_layers_2': trial.suggest_int('hidden_layers_2', 4, 32),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.01, 0.2),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                'use_bagging': trial.suggest_categorical('use_bagging', [True, False])
            }
            # Training parameters are now defined in the constructor
            model = DeepEnsembleRegressor(
                n_models=params['n_models'],
                hidden_layers=[params['hidden_layers_1'], params['hidden_layers_2']],
                dropout_rate=params['dropout_rate'],
                learning_rate=params['learning_rate'],
                batch_size=params['batch_size'],
                use_bagging=params['use_bagging'],
                epochs=100, # Use a fixed number of epochs for the trial
                early_stopping_patience=10 # Early stopping is crucial for HPO
            )
        elif model_name == 'CNN':
                params = {
                    'filters': trial.suggest_int('filters', 4, 64),
                    'kernel_size': trial.suggest_int('kernel_size', 1, 4),
                    'dense_units': trial.suggest_int('dense_units', 2, 64),
                    'lr': trial.suggest_float('lr', 1e-3, 5e-2, log=True),
                    'epochs': trial.suggest_int('epochs', 50, 100)
                }
                model = CNNRegressor(**params)
               
                # Manual CV for CNN (no parallelization)
                kf = KFold(n_splits=3, shuffle=True, random_state=42)
                scores = []
                for train_idx, val_idx in kf.split(X_train):
                    model_clone = CNNRegressor(**params)
                    model_clone.fit(X_train[train_idx], y_train[train_idx])
                    score = mean_squared_error(y_train[val_idx], model_clone.predict(X_train[val_idx]))
                    scores.append(score)
                return np.mean(scores)
        else:
            raise ValueError(f"Model {model_name} not supported for optimization")
        # Create pipeline with scaling if needed
        if models_config[model_name]['needs_scaling']:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
        else:
            pipeline = Pipeline([
                ('model', model)
            ])
        scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='neg_mean_squared_error', n_jobs=1) # Changed to n_jobs=1 to avoid multiprocessing issues
        return -np.mean(scores) # Minimize MSE
    except Exception as e:
        print(f"Error during trial execution for {model_name}: {e}")
        return float('inf') # Return a high value to indicate failure

def optimize_hyperparameters(model_name, X_train, y_train, n_trials=100, timeout=3600):
    """ Executes hyperparameter optimization using Optuna. """
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(n_startup_trials=10, seed=42),
        pruner=optuna.pruners.HyperbandPruner()
    )
    study.optimize(
        lambda trial: objective(trial, model_name, X_train, y_train),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=1 # Reduced to avoid memory issues
    )
    return study.best_params

def robust_cross_validation(model, X, y, model_name, n_splits=5, n_repeats=3):
    """ Perform repeated k-fold cross-validation for robust performance estimation """
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    scoring = {
        'mse': 'neg_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'r2': 'r2'
    }
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1, return_train_score=True, error_score='raise') # Added error_score='raise'
    results = {
        'test_mse_mean': -np.mean(cv_results['test_mse']),
        'test_mse_std': np.std(cv_results['test_mse']),
        'test_mae_mean': -np.mean(cv_results['test_mae']),
        'test_mae_std': np.std(cv_results['test_mae']),
        'test_r2_mean': np.mean(cv_results['test_r2']),
        'test_r2_std': np.std(cv_results['test_r2']),
        'n_splits': n_splits,
        'n_repeats': n_repeats
    }
    return results

def train_and_evaluate(model_name, best_params, X_train, y_train, X_test, y_test):
    """ Trains the model with the best found hyperparameters and evaluates on the test set. """
    models_config = get_model_configuration()
    if model_name == 'NGBoost':
        model = NGBRegressor(**best_params)
    elif model_name == 'CatBoost':
        model = CatBoostRegressor(**best_params, verbose=0, early_stopping_rounds=20)
    elif model_name == 'SVM':
        model = SVR(**best_params)
    elif model_name == 'ElasticNet':
        model = ElasticNet(**best_params, random_state=42)
    elif model_name == 'XGBoost':
        model = XGBRegressor(**best_params, random_state=42)
    elif model_name == 'RandomForest':
        model = RandomForestRegressor(**best_params, random_state=42)
    elif model_name == 'LightGBM':
        model = LGBMRegressor(**best_params, random_state=42)
    elif model_name == 'DeepEnsemble' and TENSORFLOW_AVAILABLE:
        model = DeepEnsembleRegressor(
            n_models=best_params.get('n_models', 5),
            hidden_layers=[best_params.get('hidden_layers_1', 64), best_params.get('hidden_layers_2', 32)],
            dropout_rate=best_params.get('dropout_rate', 0.2)
        )
    elif model_name == 'CNN':
        model = CNNRegressor(
            filters=best_params.get('filters', 32),
            kernel_size=best_params.get('kernel_size', 2),
            dense_units=best_params.get('dense_units', 32),
            lr=best_params.get('lr', 0.01),
            epochs=best_params.get('epochs', 50)
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        uncertainty_info = {}
    else:
        raise ValueError(f"Model {model_name} not supported")
    # Create pipeline with scaling if needed
    if models_config[model_name]['needs_scaling']:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
    else:
        pipeline = Pipeline([
            ('model', model)
        ])
    pipeline.fit(X_train, y_train)
    # Handle DeepEnsemble special prediction
    if model_name == 'DeepEnsemble' and TENSORFLOW_AVAILABLE:
        y_pred, total_std, epistemic_std, aleatoric_std = pipeline.named_steps['model'].predict_with_uncertainty(X_test)
        uncertainty_info = {'total_std': total_std.tolist(), 'epistemic_std': epistemic_std.tolist(), 'aleatoric_std': aleatoric_std.tolist()}
    else:
        y_pred = pipeline.predict(X_test)
        uncertainty_info = {}
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{model_name} - MSE: {mse:.4f} - R²: {r2:.4f} - MAE: {mae:.4f}")
    return pipeline, y_pred, mse, r2, mae, uncertainty_info

def perform_residual_analysis(y_true, y_pred, model_name):
    """ Comprehensive residual analysis for model diagnostics """
    residuals = y_true - y_pred
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Residuals vs Predicted
    axes[0,0].scatter(y_pred, residuals, alpha=0.6)
    axes[0,0].axhline(y=0, color='red', linestyle='--')
    axes[0,0].set_xlabel('Predicted Values')
    axes[0,0].set_ylabel('Residuals')
    axes[0,0].set_title('Residuals vs Predicted')
    axes[0,0].grid(True, alpha=0.3)
    # QQ plot for normality
    stats.probplot(residuals, dist="norm", plot=axes[0,1])
    axes[0,1].set_title('Q-Q Plot for Normality')
    axes[0,1].grid(True, alpha=0.3)
    # Residual distribution
    axes[1,0].hist(residuals, bins=30, alpha=0.7, density=True)
    axes[1,0].set_xlabel('Residuals')
    axes[1,0].set_ylabel('Density')
    axes[1,0].set_title('Residual Distribution')
    axes[1,0].grid(True, alpha=0.3)
    # Autocorrelation of residuals
    autocorrelation_plot(residuals, ax=axes[1,1])
    axes[1,1].set_title('Residual Autocorrelation')
    axes[1,1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/residual_analysis_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    # Statistical tests
    shapiro_stat, shapiro_p = shapiro(residuals)
    jb_stat, jb_p = jarque_bera(residuals)
    return {
        'residual_mean': float(np.mean(residuals)),
        'residual_std': float(np.std(residuals)),
        'residual_skew': float(stats.skew(residuals)),
        'residual_kurtosis': float(stats.kurtosis(residuals)),
        'shapiro_stat': float(shapiro_stat),
        'shapiro_p': float(shapiro_p),
        'jarque_bera_stat': float(jb_stat),
        'jarque_bera_p': float(jb_p),
        'normality_rejected': int(shapiro_p < 0.05 or jb_p < 0.05)
    }

def explain_predictions_compatible(model, X_test, y_test, feature_names, model_name):
    """ Apply XAI techniques to explain model predictions with compatibility fixes """
    if not SHAP_AVAILABLE:
        return {
            "shap_values": [],
            "feature_importance": [],
            "success": False,
            "error": "SHAP not available"
        }
    try:
        # Use TreeExplainer for tree-based models
        if model_name in ['XGBoost', 'RandomForest', 'LightGBM', 'CatBoost']:
            try:
                explainer = shap.TreeExplainer(model.named_steps['model'])
                shap_values = explainer.shap_values(X_test)
            except:
                # Fallback to KernelExplainer
                explainer = shap.KernelExplainer(model.predict, X_test)
                shap_values = explainer.shap_values(X_test)
        else:
            # Use KernelExplainer for other models
            explainer = shap.KernelExplainer(model.predict, X_test)
            shap_values = explainer.shap_values(X_test)
        # Calculate feature importance
        if hasattr(shap_values, 'shape') and len(shap_values.shape) > 1:
            feature_importance = np.abs(shap_values).mean(axis=0)
        else:
            feature_importance = np.abs(shap_values).mean()
        # Create summary plot
        plt.figure(figsize=(10, 8))
        if hasattr(shap_values, 'shape') and len(shap_values.shape) > 1:
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        else:
            # Handle 1D shap values
            plt.barh(feature_names, feature_importance)
            plt.xlabel('Mean |SHAP value|')
            plt.title(f'Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(f'results/shap_summary_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        # Create bar plot of feature importance
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(feature_names))
        if len(feature_importance) == len(feature_names):
            plt.barh(y_pos, feature_importance)
        else:
            # If feature importance is scalar, distribute equally
            plt.barh(y_pos, [feature_importance] * len(feature_names))
        plt.yticks(y_pos, feature_names)
        plt.xlabel('Mean |SHAP value|')
        plt.title(f'Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(f'results/shap_importance_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        return {
            'shap_values': shap_values.tolist() if hasattr(shap_values, 'tolist') else [],
            'feature_importance': feature_importance.tolist() if hasattr(feature_importance, 'tolist') else float(feature_importance),
            'success': True
        }
    except Exception as e:
        print(f"SHAP explanation failed for {model_name}: {e}")
        return {
            'shap_values': [],
            'feature_importance': [],
            'success': False,
            'error': str(e)
        }

def plot_results(y_test, model_name, n_trials, sample_frac, y_pred, mse, r2):
    """ Generates a scatter plot comparing real and predicted values for a single model, displaying MSE and R² metrics. """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel("Real Value")
    plt.ylabel("Predicted Value")
    plt.title(f"{model_name}\nMSE: {mse:.4f} - R²: {r2:.4f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/model_metrics_{model_name}_{n_trials}_{sample_frac}.png', dpi=300)
    plt.savefig(f'results/model_metrics_{model_name}_{n_trials}_{sample_frac}.pdf', dpi=300)
    plt.close()

def generate_final_report(results, model_name, feature_names):
    """ Generate a comprehensive final report """
    report = f"""
COMPREHENSIVE MODEL ANALYSIS REPORT
===================================
Model: {model_name}
Justification: {results['model_justification']}
PERFORMANCE METRICS:
- Test MSE: {results['test_metrics']['mse']:.4f}
- Test R²: {results['test_metrics']['r2']:.4f}
- Test MAE: {results['test_metrics']['mae']:.4f}
- Cross-validation MSE: {results['cv_results']['test_mse_mean']:.4f} ± {results['cv_results']['test_mse_std']:.4f}
- Cross-validation R²: {results['cv_results']['test_r2_mean']:.4f} ± {results['cv_results']['test_r2_std']:.4f}
FEATURE ANALYSIS:
"""
    for feature in feature_names:
        corr = results['feature_analysis']['correlations'].get(feature, 0)
        imp = results['feature_analysis']['importance'].get(feature, 0)
        report += f"- {feature}: Correlation={corr:.3f}, Importance={imp:.3f}\n"
    report += f"""
RESIDUAL ANALYSIS:
- Residual mean: {results['residual_analysis']['residual_mean']:.4f}
- Residual std: {results['residual_analysis']['residual_std']:.4f}
- Residual skew: {results['residual_analysis']['residual_skew']:.4f}
- Normality rejected: {results['residual_analysis']['normality_rejected']}
MODEL INTERPRETATION:
"""
    # Sort features by importance
    if (results.get('xai_analysis', {}).get('success', False) and 'feature_importance' in results.get('xai_analysis', {})):
        xai_feature_importance = results['xai_analysis']['feature_importance']
        if isinstance(xai_feature_importance, list) and len(xai_feature_importance) == len(feature_names):
            importance_dict = dict(zip(feature_names, xai_feature_importance))
        else:
            # If it's a single value, use RF importance instead
            importance_dict = results['feature_analysis']['importance']
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        report += "Key drivers of water quality predictions:\n"
        for feature, importance in sorted_features:
            report += f"- {feature}: importance = {importance:.3f}\n"
    else:
        # Use RF importance if XAI failed
        importance_dict = results['feature_analysis']['importance']
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        report += "Key drivers of water quality predictions (RF importance):\n"
        for feature, importance in sorted_features:
            report += f"- {feature}: importance = {importance:.3f}\n"
    report += f"""
HYPERPARAMETERS: {json.dumps(results['best_params'], indent=2)}
EXPERIMENTAL SETUP:
- Number of trials: {results.get('n_trials', 'N/A')}
- Sample fraction: {results.get('sample_frac', 'N/A')}
- Random seed: {results.get('seed', 'N/A')}
"""
    with open(f"results/analysis_report_{model_name}.txt", "w") as f:
        f.write(report)
    return report

def enhanced_main(model_name, n_trials, timeout, sample_frac, seed):
    """Enhanced main function with comprehensive analysis"""
   
    # Setup
    os.makedirs("results", exist_ok=True)
    log_file = f"results/experiment_{model_name}_{n_trials}_{sample_frac}_{seed}.log"
   
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        filename=log_file,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
   
    #  TOTAL EXPERIMENT TIME START
    total_start_time = time.time()
   
    logging.info(f"Starting comprehensive experiment for {model_name}")
   
    # Read and prepare data
    X_train, X_test, y_train, y_test = read_wq_taiwan(seed=seed)
    feature_names = ['EC', 'SS', 'WT', 'pH']
   
    # Apply sampling if needed
    if sample_frac < 1.0:
        idx = np.random.choice(len(X_train), int(len(X_train)*sample_frac), replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
        logging.info(f"Using {sample_frac*100:.1f}% of training data.")
   
    
    # Model configuration with justification
    models_config = get_model_configuration()
    if model_name not in models_config:
        raise ValueError(f"Model {model_name} not supported")
   
    logging.info(f"Model justification: {models_config[model_name]['justification']}")
   
    # Robust cross-validation
    logging.info("Performing robust cross-validation...")
    base_model = models_config[model_name]['model']
   
    if models_config[model_name]['needs_scaling']:
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', base_model)])
    else:
        pipeline = Pipeline([('model', base_model)])
   
    cv_results = robust_cross_validation(pipeline, X_train, y_train, model_name)
    logging.info(f"CV Results - MSE: {cv_results['test_mse_mean']:.4f} ± {cv_results['test_mse_std']:.4f}")
   
    #  HYPERPARAMETER OPTIMIZATION TIME START
    logging.info("Starting hyperparameter optimization...")
    hpo_start_time = time.time()
   
    best_params = optimize_hyperparameters(model_name, X_train, y_train, n_trials, timeout)
   
    hpo_time = time.time() - hpo_start_time #  HPO TIME STOP
    logging.info(f"Best parameters for {model_name}: {best_params}")
    logging.info(f" HPO took {hpo_time:.2f} seconds")
   
    #  TRAINING TIME START
    logging.info("Training final model...")
    training_start_time = time.time()
   
    model_pipeline, y_pred, mse, r2, mae, uncertainty_info = train_and_evaluate(
        model_name, best_params, X_train, y_train, X_test, y_test
    )
   
    training_time = time.time() - training_start_time #  TRAINING TIME STOP
    logging.info(f" Training took {training_time:.2f} seconds")
   
    # Residual analysis
    logging.info("Performing residual analysis...")
    residual_results = perform_residual_analysis(y_test, y_pred, model_name)
   
    # XAI explanation
    logging.info("Generating XAI explanations...")
    xai_results = explain_predictions_compatible(
        model_pipeline, X_test, y_test, feature_names, model_name
    )
   
    # Feature importance analysis
    logging.info("Computing feature importance...")
    from sklearn.inspection import permutation_importance
   
    # Feature analysis
    logging.info("Performing feature analysis...")
    perm_importance = permutation_importance(
        model_pipeline, X_test, y_test,
        scoring='neg_mean_squared_error',
        n_repeats=50,
        random_state=seed,
    )
    feature_importances = dict(zip(feature_names, perm_importance.importances_mean))
    print(f"\n📊 Feature Importance:\n{feature_importances}")
   
    #  TOTAL EXPERIMENT TIME STOP
    total_time = time.time() - total_start_time
   
    # Save comprehensive results with timing
    results = {
        "model": model_name,
        "model_justification": models_config[model_name]['justification'],
        "cv_results": cv_results,
        "test_metrics": {"mse": mse, "r2": r2, "mae": mae},
        "residual_analysis": residual_results,
        "feature_analysis": {
            "correlations": feature_results['correlation_with_target'].to_dict(),
            "importance": feature_importances,
        },
        "xai_analysis": xai_results,
        "best_params": best_params,
        "uncertainty_info": uncertainty_info,
        "n_trials": n_trials,
        "sample_frac": sample_frac,
        "seed": seed,
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        'feature_names': feature_names,
        'feature_importance': feature_importances,
       
        #  TIMING METRICS
        'timing': {
            'hpo_seconds': float(hpo_time),
            'training_seconds': float(training_time),
            'total_seconds': float(total_time),
            'hpo_minutes': round(hpo_time / 60, 2),
            'training_minutes': round(training_time / 60, 2),
            'total_minutes': round(total_time / 60, 2)
        }
    }
   
    with open(f"results/comprehensive_results_{model_name}_{seed}.json", "w") as f:
        json.dump(results, f, indent=2)
   
    # Generate final report
    report = generate_final_report(results, model_name, feature_names)
   
    # Plot results
    plot_results(y_test, model_name, n_trials, sample_frac, y_pred, mse, r2)
   
    print(f"\n{'='*60}")
    print(f"TIMING SUMMARY - {model_name}")
    print(f"{'='*60}")
    print(f"HPO Time: {hpo_time:>8.2f}s ({hpo_time/60:>6.2f} min)")
    print(f"Training Time: {training_time:>8.2f}s ({training_time/60:>6.2f} min)")
    print(f"Total Time: {total_time:>8.2f}s ({total_time/60:>6.2f} min)")
    print(f"{'='*60}\n")
   
    logging.info(f"Experiment completed for {model_name}")
    logging.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
   
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enhanced ML experiments for water quality analysis")
    parser.add_argument("--model", type=str, default='XGBoost', choices=['NGBoost', 'CatBoost', 'SVM', 'ElasticNet', 'XGBoost', 'RandomForest', 'LightGBM', 'LinearRegression', 'CNN',], help="Name of the model to execute")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of Optuna optimization trials")
    parser.add_argument("--timeout", type=int, default=3600, help="Maximum time (seconds) for optimization")
    parser.add_argument("--sample_frac", type=float, default=1.0, help="Fraction of training data to use (0.0 to 1.0)")
    parser.add_argument("--seed", type=int, default=52, help="Seed for reproducibility")
    parser.add_argument("--full_analysis", action='store_true', help="Perform comprehensive analysis including XAI and feature analysis")
    parser.add_argument("--n_runs", type=int, default=30, help="Number of runs with different random seeds")
    args = parser.parse_args()
    # Run experiments with different seeds
    for i in range(args.n_runs):
        seed = args.seed + i
        print(f"Run {i+1}/{args.n_runs} with seed {seed}")
        if args.full_analysis:
            enhanced_main(args.model, args.n_trials, args.timeout, args.sample_frac, seed)
        else:
            break
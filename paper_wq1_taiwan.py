'''
ADAPTAÇÕES:
[X] Alterações que ele mesmo fez
[X] def read_wq_taiwan(test_size=0.25, seed=42) -> colocar a seeds como parametro
[X] Remoção de outliers -> não fazer
[X] kf = KFold(n_splits=5, shuffle=True, random_state=42) -> colocar a seeds como parametro
[X] def optimize_hyperparameters(model_name, X_train, y_train, n_trials=50, timeout=3600):
    [X] aumentar o n_trials
    [X] salvar y_true e o y_pred para calcular estatisticas depois
    [X] salvar em json inves de csv
    [X] incluir xgboost para comparação justa com o ngboost
    [X] 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
[X] Aumentar o span para 0.01, 1 e se necessario coloque log=True
[X] Rodar para cada modelo pelo menos 30 vezes com diferentes sementes de números aleatórios
[] Também fazer um "resumo gráfico" da proposta para termos um overview
[X] Fixar os conjuntos treino/teste
'''

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

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from ngboost import NGBRegressor
import optuna


def read_wq_taiwan(test_size=0.25, seed=42):
    """
    Leitura e pré-processamento dos dados de qualidade de água.
    """
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

def objective(trial, model_name, X_train, y_train):
    """
    Função objetivo para otimização dos hiperparâmetros com Optuna.
    Realiza validação cruzada (5-fold) e minimiza o MSE.
    """
    try:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        if model_name == 'NGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'minibatch_frac': trial.suggest_float('minibatch_frac', 0.5, 1.0),
                'natural_gradient': trial.suggest_categorical('natural_gradient', [True, False]),
            }
            model = NGBRegressor(**params)
        elif model_name == 'CatBoost':
            params = {
                'iterations': trial.suggest_int('iterations', 100, 300),
                'depth': trial.suggest_int('depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            }
            model = CatBoostRegressor(**params, verbose=0, early_stopping_rounds=20)
        elif model_name == 'SVM':
            params = {
                'C': trial.suggest_loguniform('C', 1e-2, 1e2),
                'epsilon': trial.suggest_float('epsilon', 0.001, 0.1),
                'kernel':  trial.suggest_categorical('kernel', ['linear', 'rbf']),
            }
            model = SVR(**params)
        elif model_name == 'ElasticNet':
            params = {
                'alpha': trial.suggest_loguniform('alpha', 1e-4, 1e0),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 1.0),
            }
            model = ElasticNet(**params)
        elif model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 1.0),
            }
            model = XGBRegressor(**params)
        else:
            raise ValueError("Modelo não suportado")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        scores = cross_val_score(pipeline, X_train, y_train, cv=kf,
                                scoring='neg_mean_squared_error', n_jobs=-1)
        return -np.mean(scores)  # Minimizar MSE

    except Exception as e:
        print(f"Erro durante a execução do trial: {e}")
        return float('inf')  # Retorna um valor alto para indicar falha

def optimize_hyperparameters(model_name, X_train, y_train, n_trials=100, timeout=3600):
    """
    Executa a otimização dos hiperparâmetros utilizando Optuna.
    """
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
        pruner=optuna.pruners.HyperbandPruner()
    )

    study.optimize(
        lambda trial: objective(trial, model_name, X_train, y_train),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=-1  # Paralelização
    )

    return study.best_params

def train_and_evaluate(model_name, best_params, X_train, y_train, X_test, y_test):
    """
    Treina o modelo com os melhores hiperparâmetros encontrados e avalia no conjunto de teste.
    Utiliza pipeline com StandardScaler.
    """
    if model_name == 'NGBoost':
        model = NGBRegressor(**best_params)
    elif model_name == 'CatBoost':
        model = CatBoostRegressor(**best_params, verbose=0, early_stopping_rounds=20)
    elif model_name == 'SVM':
        model = SVR(**best_params)
    elif model_name == 'ElasticNet':
        model = ElasticNet(**best_params)
    elif model_name == 'XGBoost':
        model = XGBRegressor(**best_params)
    else:
        raise ValueError("Modelo não suportado")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - MSE: {mse:.4f} - R²: {r2:.4f}")
    return pipeline, y_pred, mse, r2

def plot_results(y_test, model_name, n_trials, sample_frac, y_pred, mse, r2):
    """
    Gera um gráfico de dispersão comparando os valores reais e previstos para um único modelo,
    exibindo as métricas MSE e R².
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel("Valor Real")
    plt.ylabel("Valor Predito")
    plt.title(f"{model_name}\nMSE: {mse:.2f} - R²: {r2:.2f}")
    plt.tight_layout()
    plt.savefig(f'cd/model_metrics_{model_name}_{n_trials}_{sample_frac}.png', dpi=300)
    plt.savefig(f'cd/model_metrics_{model_name}_{n_trials}_{sample_frac}.pdf', dpi=300)

def main(model_name, n_trials, timeout, sample_frac, seed):
    # Configuração do logging
    os.system(f"mkdir -p cd/")
    os.system(f"touch    cd/experiment_{model_name}_{n_trials}_{sample_frac}_{seed}.log")
    logging.basicConfig(
        level=logging.INFO,
        filename=f"cd/experiment_{model_name}_{n_trials}_{sample_frac}_{seed}.log",
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info(f"Iniciando experimento para {model_name}")

    # Leitura dos dados
    X_train, X_test, y_train, y_test = read_wq_taiwan(seed=42)

    if sample_frac < 1.0:
        idx = np.random.choice(len(X_train), int(len(X_train)*sample_frac), replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
        logging.info(f"Usando {sample_frac*100:.1f}% dos dados de treino.")

    start_time = time.time()
    best_params = optimize_hyperparameters(model_name, X_train, y_train,
                                          n_trials=n_trials, timeout=timeout)
    logging.info(f"Melhores parâmetros para {model_name}: {best_params}")

    model_pipeline, y_pred, mse, r2 = train_and_evaluate(model_name, best_params, X_train, y_train, X_test, y_test)
    end_time = time.time()
    elapsed_time = end_time - start_time

    logging.info(f"{model_name} - MSE: {mse:.4f} - R²: {r2:.4f}, Tempo total: {elapsed_time:.2f} s")
    print(f"Tempo total de execução para {model_name}: {elapsed_time:.2f} s")

    # Salva os resultados em JSON
    results = {
        "model": model_name,
        "mse": mse,
        "r2": r2,
        "execution_time": elapsed_time,
        "best_params": best_params,
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist()
    }
    with open(f"cd/results_{model_name}_{n_trials}_{sample_frac}_{seed}.json", "w") as f:
        json.dump(results, f)

    # Geração dos gráficos dos resultados
    plot_results(y_test, model_name, n_trials, sample_frac, y_pred, mse, r2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Execução de experimentos de ML para análise de qualidade de água")
    parser.add_argument("--model", type=str, 
                        default='ElasticNet',
                        choices=['NGBoost', 'CatBoost', 'SVM', 'ElasticNet', 'XGBoost'],
                        help="Nome do modelo a ser executado")
    parser.add_argument("--n_trials", type=int, default=100,
                        help="Número de tentativas de otimização com Optuna")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Tempo máximo (em segundos) para otimização")
    parser.add_argument("--sample_frac", type=float, default=1.0,
                        help="Fração dos dados de treinamento a ser usada (0.0 a 1.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semente para reproducibilidade")
    args = parser.parse_args()

    # Rodar o experimento 30 vezes com diferentes sementes
    for i in range(30):
        seed = args.seed + i
        main(args.model, args.n_trials, args.timeout, args.sample_frac, seed)

    
# 'n_estimators': trial.suggest_int('n_estimators', 50, 300),
# 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
# 'minibatch_frac': trial.suggest_float('minibatch_frac', 0.1, 1.0),
# 'natural_gradient': trial.suggest_categorical('natural_gradient', [True, False]),

# lamap@lamap-XPS-13-7390:~/Downloads$ python3.10 cd_a2.py --model NGBoost --n_trials 20 --sample_frac 0.75
# NGBoost - MSE: 0.4286 - R²: 0.7641
# Tempo total de execução para NGBoost: 33.92 s

# lamap@lamap-XPS-13-7390:~/Downloads$ python3.10 cd_a2.py --model NGBoost --n_trials 30 --sample_frac 0.75
# NGBoost - MSE: 0.4828 - R²: 0.7342
# Tempo total de execução para NGBoost: 39.67 s

# lamap@lamap-XPS-13-7390:~/Downloads$ python3.10 cd_a2.py --model NGBoost --n_trials 30 --sample_frac 0.9
# NGBoost - MSE: 0.3907 - R²: 0.7849
# Tempo total de execução para NGBoost: 35.68 s

# lamap@lamap-XPS-13-7390:~/Downloads$ python3.10 cd_a2.py --model NGBoost --n_trials 50 --sample_frac 0.9
# NGBoost - MSE: 0.3857 - R²: 0.7877
# Tempo total de execução para NGBoost: 110.11 s

# lamap@lamap-XPS-13-7390:~/Downloads$ python3.10 cd_a2.py --model NGBoost --n_trials 100 --sample_frac 0.9
# NGBoost - MSE: 0.4171 - R²: 0.7704
# Tempo total de execução para NGBoost: 197.20 s

# lamap@lamap-XPS-13-7390:~/Downloads$ python3.10 cd_a2.py --model NGBoost --n_trials 100 --sample_frac 0.8
# NGBoost - MSE: 0.4352 - R²: 0.7604
# Tempo total de execução para NGBoost: 166.21 s

# lamap@lamap-XPS-13-7390:~/Downloads$ python3.10 cd_a2.py --model NGBoost --n_trials 100 --sample_frac 0.95
# NGBoost - MSE: 0.4147 - R²: 0.7717
# Tempo total de execução para NGBoost: 223.42 s

# lamap@lamap-XPS-13-7390:~/Downloads$ python3.10 cd_a2.py --model NGBoost --n_trials 50 --sample_frac 0.95
# NGBoost - MSE: 0.3958 - R²: 0.7821
# Tempo total de execução para NGBoost: 62.46 s

# lamap@lamap-XPS-13-7390:~/Downloads$ python3.10 cd_a2.py --model NGBoost --n_trials 30 --sample_frac 0.95
# NGBoost - MSE: 0.3653 - R²: 0.7989
# Tempo total de execução para NGBoost: 89.06 s

# lamap@lamap-XPS-13-7390:~/Downloads$ python3.10 cd_a2.py --model NGBoost --n_trials 25 --sample_frac 0.95
# NGBoost - MSE: 0.4101 - R²: 0.7742
# Tempo total de execução para NGBoost: 41.10 s

## lamap@lamap-XPS-13-7390:~/Downloads$ python3.10 cd_a2.py --model NGBoost --n_trials 40 --sample_frac 0.95
## NGBoost - MSE: 0.3538 - R²: 0.8052
## Tempo total de execução para NGBoost: 126.07 s






# 'n_estimators': trial.suggest_int('n_estimators', 100, 300),
# 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
# 'minibatch_frac': trial.suggest_float('minibatch_frac', 0.5, 1.0),
# 'natural_gradient': trial.suggest_categorical('natural_gradient', [True]),

# lamap@lamap-XPS-13-7390:~/Downloads$ python3.10 cd_a2.py --model NGBoost --n_trials 40 --sample_frac 0.95
# NGBoost - MSE: 0.4041 - R²: 0.7775
# Tempo total de execução para NGBoost: 106.30 s

# 'n_estimators': trial.suggest_int('n_estimators', 100, 500),
# 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
# 'minibatch_frac': trial.suggest_float('minibatch_frac', 0.1, 1.0),
# 'natural_gradient': trial.suggest_categorical('natural_gradient', [True]),

# lamap@lamap-XPS-13-7390:~/Downloads$ python3.10 cd_a2.py --model NGBoost --n_trials 40 --sample_frac 0.95
# NGBoost - MSE: 0.3943 - R²: 0.7829
# Tempo total de execução para NGBoost: 175.45 s

# 'n_estimators': trial.suggest_int('n_estimators', 50, 200),
# 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
# 'minibatch_frac': trial.suggest_float('minibatch_frac', 0.1, 1.0),
# 'natural_gradient': trial.suggest_categorical('natural_gradient', [True]),

# lamap@lamap-XPS-13-7390:~/Downloads$ python3.10 cd_a2.py --model NGBoost --n_trials 40 --sample_frac 0.95
# NGBoost - MSE: 0.3888 - R²: 0.7860
# Tempo total de execução para NGBoost: 66.41 s

# 'n_estimators': trial.suggest_int('n_estimators', 50, 300),
# 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
# 'minibatch_frac': trial.suggest_float('minibatch_frac', 0.1, 1.0),
# 'natural_gradient': trial.suggest_categorical('natural_gradient', [True, False]),

# lamap@lamap-XPS-13-7390:~/Downloads$ python3.10 cd_a2.py --model NGBoost --n_trials 40 --sample_frac 0.95
# NGBoost - MSE: 0.4427 - R²: 0.7563
# Tempo total de execução para NGBoost: 66.74 s

# 'n_estimators': trial.suggest_int('n_estimators', 300, 500),
# 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
# 'minibatch_frac': trial.suggest_float('minibatch_frac', 0.1, 1.0),
# 'natural_gradient': trial.suggest_categorical('natural_gradient', [True, False]),

# lamap@lamap-XPS-13-7390:~/Downloads$ python3.10 cd_a2.py --model NGBoost --n_trials 40 --sample_frac 0.95
# NGBoost - MSE: 0.3963 - R²: 0.7819
# Tempo total de execução para NGBoost: 252.49 s

# 'n_estimators': trial.suggest_int('n_estimators', 500, 1000),
# 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
# 'minibatch_frac': trial.suggest_float('minibatch_frac', 0.1, 1.0),
# 'natural_gradient': trial.suggest_categorical('natural_gradient', [True, False]),

# lamap@lamap-XPS-13-7390:~/Downloads$ python3.10 cd_a2.py --model NGBoost --n_trials 40 --sample_frac 0.95
# NGBoost - MSE: 0.4028 - R²: 0.7783
# Tempo total de execução para NGBoost: 548.74 s


### RESULTADOS APÓS MODIFICAÇÕES



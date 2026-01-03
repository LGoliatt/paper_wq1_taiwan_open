#!/bin/bash

python3 paper_wq1_taiwan_review.py --model ElasticNet       --n_runs 30 --full_analysis --n_trials 50
python3 paper_wq1_taiwan_review.py --model SVM              --n_runs 30 --full_analysis --n_trials 50
python3 paper_wq1_taiwan_review.py --model XGBoost          --n_runs 30 --full_analysis --n_trials 50
python3 paper_wq1_taiwan_review.py --model NGBoost          --n_runs 30 --full_analysis --n_trials 50
python3 paper_wq1_taiwan_review.py --model CatBoost         --n_runs 30 --full_analysis --n_trials 50
python3 paper_wq1_taiwan_review.py --model CNN              --n_runs 30 --full_analysis --n_trials 50
#--
#python3 paper_wq1_taiwan_review.py --model LightGBM         --n_runs 30 --full_analysis --n_trials 50
#python3 paper_wq1_taiwan_review.py --model LinearRegression --n_runs 30 --full_analysis --n_trials 50
#python3 paper_wq1_taiwan_review.py --model RandomForest     --n_runs 30 --full_analysis --n_trials 50



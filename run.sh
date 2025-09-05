# Example: Run NGBoost 30 times
python wq_taiwan_experiment.py --model NGBoost --n_trials 50 --sample_frac 0.95 --n_runs 30

# Compare all models
for model in NGBoost XGBoost CatBoost SVM ElasticNet; do
    python wq_taiwan_experiment.py --model $model --n_trials 50 --n_runs 30
done

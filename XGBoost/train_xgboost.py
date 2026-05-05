import os
import sys
import argparse
import json
import time

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor
import joblib

# ensure local imports work when running from project root
sys.path.append(os.path.dirname(__file__))
from featurizer import ABX3Featurizer


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost for ABX3 band gap prediction')
    parser.add_argument('--data', required=True, help='CSV file with dataset')
    parser.add_argument('--target', default='band_gap', help='Target column name')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--model_out', default='XGBoost/model.joblib')
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in data.")

    y = df[args.target].values

    featurizer = ABX3Featurizer()
    X_df = featurizer.featurize(df)
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df.fillna(X_df.mean(), inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=args.test_size, random_state=args.random_state)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBRegressor(objective='reg:squarederror', random_state=args.random_state, n_jobs=args.n_jobs))
    ])

    param_grid = {
        'xgb__n_estimators': [100, 300],
        'xgb__max_depth': [3, 6],
        'xgb__learning_rate': [0.01, 0.1],
        'xgb__subsample': [0.8, 1.0],
        'xgb__colsample_bytree': [0.8, 1.0],
    }

    grid = GridSearchCV(pipeline, param_grid, cv=args.cv, scoring='neg_mean_squared_error', n_jobs=args.n_jobs, verbose=1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    os.makedirs(os.path.dirname(args.model_out) or '.', exist_ok=True)
    joblib.dump(best, args.model_out)

    metrics = {
        'rmse': float(rmse),
        'r2': float(r2),
        'best_params': grid.best_params_,
        'cv_best_score_neg_mse': float(grid.best_score_)
    }
    metrics_path = os.path.splitext(args.model_out)[0] + '_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to {args.model_out}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == '__main__':
    main()

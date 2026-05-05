import os
import sys
import argparse

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor
import joblib

sys.path.append(os.path.dirname(__file__))
from featurizer import ABX3Featurizer


def main():
    parser = argparse.ArgumentParser(description='Quick smoke training for XGBoost (ABX3)')
    parser.add_argument('--data', required=True, help='CSV file with dataset')
    parser.add_argument('--target', default='band_gap', help='Target column name')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--n_jobs', type=int, default=2)
    parser.add_argument('--n_estimators', type=int, default=50)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--model_out', default='XGBoost/smoke_model.joblib')
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
        ('xgb', XGBRegressor(objective='reg:squarederror', random_state=args.random_state, n_jobs=args.n_jobs,
                             n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.learning_rate))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    os.makedirs(os.path.dirname(args.model_out) or '.', exist_ok=True)
    joblib.dump(pipeline, args.model_out)

    print(f"Smoke training complete. RMSE={rmse:.4f}, R2={r2:.4f}")
    metrics_path = os.path.splitext(args.model_out)[0] + '_metrics.json'
    try:
        import json
        metrics = {'rmse': float(rmse), 'r2': float(r2)}
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
    except Exception:
        pass


if __name__ == '__main__':
    main()

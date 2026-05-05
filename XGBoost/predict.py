import os
import sys
import argparse

import pandas as pd

sys.path.append(os.path.dirname(__file__))
from featurizer import ABX3Featurizer
import joblib


def main():
    parser = argparse.ArgumentParser(description='Predict with trained XGBoost model')
    parser.add_argument('--model', required=True, help='Trained model file (joblib)')
    parser.add_argument('--input', required=True, help='Input CSV with A/B/X or formula')
    parser.add_argument('--output', default='XGBoost/predictions.csv', help='Output CSV with predictions')
    parser.add_argument('--target', default=None, help='(Optional) target column name if present')
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    featurizer = ABX3Featurizer()
    X_df = featurizer.featurize(df)
    X_df.fillna(X_df.mean(), inplace=True)

    model = joblib.load(args.model)
    preds = model.predict(X_df)

    out = df.copy()
    out['pred_band_gap'] = preds
    out.to_csv(args.output, index=False)
    print(f'Predictions saved to {args.output}')


if __name__ == '__main__':
    main()

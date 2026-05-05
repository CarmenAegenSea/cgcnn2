import os
import sys
import argparse

import pandas as pd

sys.path.append(os.path.dirname(__file__))
from featurizer import ABX3Featurizer
import joblib
import numbers


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

    model_obj = joblib.load(args.model)

    # backward compatible: model may be a plain pipeline or a dict containing pipeline + metadata
    if isinstance(model_obj, dict) and 'pipeline' in model_obj:
        pipeline = model_obj['pipeline']
        scale_target = bool(model_obj.get('scale_target', False))
        y_mean = float(model_obj.get('y_mean', 0.0))
        y_std = float(model_obj.get('y_std', 1.0))
    else:
        pipeline = model_obj
        scale_target = False

    preds = pipeline.predict(X_df)
    if scale_target:
        preds = preds * y_std + y_mean

    out = df.copy()
    col_name = 'pred_band_gap' if args.target is None else f'pred_{args.target}'
    out[col_name] = preds
    out.to_csv(args.output, index=False)
    print(f'Predictions saved to {args.output}')

if __name__ == '__main__':
    main()

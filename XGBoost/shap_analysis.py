"""SHAP analysis helper for XGBoost models.

Usage examples:
  python shap_analysis.py --model XGBoost/model_min2.joblib --data XGBoost/clean_catalysis_min2.csv --output-dir XGBoost/shap_output --sample 200
  python shap_analysis.py --data XGBoost/clean_catalysis_min2.csv --dry-run

The script provides a lightweight `--dry-run` that only featurizes the input
and reports shapes (useful when `shap` or `xgboost` are not installed).
Full SHAP computation requires `shap` and the model's Python dependencies.
"""
import os
import sys
import argparse
import json

import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(__file__))
from featurizer import ABX3Featurizer
import joblib

from sklearn.pipeline import Pipeline as SkPipeline

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set()
except Exception:
    sns = None

try:
    import shap
    _HAS_SHAP = True
except Exception:
    shap = None
    _HAS_SHAP = False


def load_model(model_path):
    """Load saved model (joblib). Returns pipeline, scale_target, y_mean, y_std."""
    obj = joblib.load(model_path)
    if isinstance(obj, dict) and 'pipeline' in obj:
        pipeline = obj['pipeline']
        scale_target = bool(obj.get('scale_target', False))
        y_mean = float(obj.get('y_mean', 0.0))
        y_std = float(obj.get('y_std', 1.0))
    else:
        pipeline = obj
        scale_target = False
        y_mean = 0.0
        y_std = 1.0
    return pipeline, scale_target, y_mean, y_std


def split_pipeline(pipeline):
    """Split an sklearn Pipeline into (preprocessor, estimator).

    If the provided object is not a Pipeline, returns (None, pipeline).
    """
    if isinstance(pipeline, SkPipeline):
        steps = pipeline.steps
        if len(steps) >= 2:
            preproc = SkPipeline(steps[:-1])
            estimator = steps[-1][1]
        else:
            preproc = None
            estimator = steps[-1][1]
    else:
        preproc = None
        estimator = pipeline
    return preproc, estimator


def prepare_X(df):
    featurizer = ABX3Featurizer()
    X_df = featurizer.featurize(df)
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df.fillna(X_df.mean(), inplace=True)
    return X_df


def basic_check(data_path):
    """Lightweight check: featurize and report feature shape and names.

    Does not require shap or xgboost. Useful as a smoke test in minimal envs.
    """
    df = pd.read_csv(data_path)
    X_df = prepare_X(df)
    return {
        'n_rows': int(X_df.shape[0]),
        'n_features': int(X_df.shape[1]),
        'feature_sample': list(X_df.columns[:20])
    }


def compute_shap(model_path, data_path, output_dir='XGBoost/shap_output', sample=None, explainer_type='tree'):
    """Compute SHAP values and save plots/CSVs to `output_dir`.

    Requires `shap` to be installed. Returns a small result dict.
    """
    if not _HAS_SHAP:
        raise ImportError('shap is not installed. Install with `pip install shap`.')

    pipeline, scale_target, y_mean, y_std = load_model(model_path)
    preproc, estimator = split_pipeline(pipeline)

    df = pd.read_csv(data_path)
    X_df = prepare_X(df)
    if sample is not None and sample > 0 and sample < len(X_df):
        X_df_sample = X_df.sample(n=sample, random_state=42)
    else:
        X_df_sample = X_df

    # Transform with preprocessing steps (if any) so explainer sees the same inputs
    if preproc is not None:
        try:
            X_trans = preproc.transform(X_df_sample)
        except Exception:
            # fallback: attempt fit_transform if transform fails (uncommon for fitted pipelines)
            X_trans = preproc.fit_transform(X_df_sample)
    else:
        X_trans = X_df_sample.values

    if hasattr(X_trans, 'toarray'):
        X_trans_arr = X_trans.toarray()
    else:
        X_trans_arr = np.asarray(X_trans)

    # choose explainer API depending on shap version
    if explainer_type == 'tree' and hasattr(shap, 'TreeExplainer'):
        expl = shap.TreeExplainer(estimator)
        try:
            shap_values = expl.shap_values(X_trans_arr)
        except Exception:
            shap_values = expl(X_trans_arr).values
    else:
        expl = shap.Explainer(estimator, X_trans_arr)
        shap_values = expl(X_trans_arr).values

    # shap_values may be a list (multi-output) or array
    if isinstance(shap_values, list):
        shap_arr = np.array(shap_values[0])
    else:
        shap_arr = np.array(shap_values)

    feature_names = list(X_df_sample.columns)
    mean_abs = np.mean(np.abs(shap_arr), axis=0)
    imp_df = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs})
    imp_df_sorted = imp_df.sort_values('mean_abs_shap', ascending=False)

    os.makedirs(output_dir, exist_ok=True)
    imp_df_sorted.to_csv(os.path.join(output_dir, 'shap_feature_importance.csv'), index=False)

    # barplot (top features)
    try:
        plt.figure(figsize=(8, 6))
        sns.barplot(x='mean_abs_shap', y='feature', data=imp_df_sorted.head(30))
        plt.title('Mean |SHAP value| (top features)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_feature_importance_bar.png'), dpi=200)
        plt.close()
    except Exception as e:
        print('Barplot failed:', e)

    # beeswarm / summary plot
    try:
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_arr, X_df_sample, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_beeswarm.png'), dpi=200)
        plt.close()
    except Exception as e:
        print('SHAP summary_plot failed:', e)

    # save shap values matrix and transformed features
    shap_df = pd.DataFrame(shap_arr, columns=feature_names, index=X_df_sample.index)
    shap_df.to_csv(os.path.join(output_dir, 'shap_values.csv'), index=True)

    X_trans_df = pd.DataFrame(X_trans_arr, columns=feature_names, index=X_df_sample.index)
    X_trans_df.to_csv(os.path.join(output_dir, 'transformed_features.csv'), index=True)

    return {'n_samples': int(shap_arr.shape[0]), 'n_features': int(shap_arr.shape[1]), 'output_dir': output_dir}


def main():
    parser = argparse.ArgumentParser(description='Compute SHAP explanations for a trained XGBoost model')
    parser.add_argument('--model', help='Trained model (joblib)')
    parser.add_argument('--data', required=True, help='Input CSV with formulas (same format as used for training)')
    parser.add_argument('--output-dir', default='XGBoost/shap_output')
    parser.add_argument('--sample', type=int, default=200, help='Number of rows to sample for SHAP (set 0 or omit to use all)')
    parser.add_argument('--explainer', choices=['tree', 'auto'], default='tree')
    parser.add_argument('--dry-run', action='store_true', help='Only featurize and report shapes (no SHAP computation)')
    args = parser.parse_args()

    if args.dry_run:
        info = basic_check(args.data)
        print(json.dumps(info, indent=2))
        return

    if args.model is None:
        print('Error: --model is required for full SHAP computation', file=sys.stderr)
        sys.exit(2)

    if not _HAS_SHAP:
        print('Error: shap package not installed. Install: pip install shap', file=sys.stderr)
        sys.exit(2)

    result = compute_shap(args.model, args.data, output_dir=args.output_dir, sample=(None if args.sample == 0 else args.sample), explainer_type=args.explainer)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

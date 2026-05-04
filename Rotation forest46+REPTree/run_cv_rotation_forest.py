import os
import sys
import argparse
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib

# ensure local folder on path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from rotation_forest import RotationForestRegressor, save_model


def choose_target_column(df, prefer=None):
    cols = list(df.columns)
    if prefer and prefer in cols:
        return prefer
    candidates = ['formation_energy_per_atom', 'band_gap', 'energy_above_hull', 'target', 'y']
    for c in candidates:
        if c in cols:
            return c
    # otherwise pick last numeric column
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        return None
    # avoid id-like columns
    for c in reversed(num_cols):
        if 'id' not in c.lower():
            return c
    return num_cols[-1]


def run_cv(source_path, local_copy_path, target_col, n_estimators=46, K=3, sample_percent=0.75, folds=5, random_state=0):
    # copy into local data folder
    os.makedirs(os.path.dirname(local_copy_path), exist_ok=True)
    if os.path.abspath(source_path) != os.path.abspath(local_copy_path):
        shutil.copy2(source_path, local_copy_path)

    df = pd.read_csv(local_copy_path)
    if target_col is None:
        target_col = choose_target_column(df)
    if target_col is None or target_col not in df.columns:
        raise ValueError('无法识别目标列，请用 --target 指定 CSV 中的目标列。')

    df = df.copy()
    # select numeric features
    numeric = df.select_dtypes(include=[np.number])
    if target_col not in numeric.columns:
        # try to coerce
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        numeric = df.select_dtypes(include=[np.number])

    if target_col not in numeric.columns:
        raise ValueError('目标列不是数值类型，请先处理数据。')

    X_all = numeric.drop(columns=[target_col]).values
    y_all = numeric[target_col].values

    # remove samples with NaN
    mask = ~np.isnan(y_all)
    mask = mask & (~np.isnan(X_all).any(axis=1))
    X_all = X_all[mask]
    y_all = y_all[mask]

    out_base = os.path.join(BASE_DIR, 'experiments', 'cv')
    outputs_dir = os.path.join(out_base, 'outputs')
    models_dir = os.path.join(out_base, 'models')
    logs_dir = os.path.join(out_base, 'logs')
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    fold_metrics = []
    all_true = []
    all_pred = []

    for i, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        model = RotationForestRegressor(n_estimators=n_estimators, K=K, sample_percent=sample_percent, random_state=random_state+i)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        fold_metrics.append({'fold': i+1, 'r2': float(r2), 'rmse': float(rmse), 'mae': float(mae)})

        # save per-fold predictions
        pd.DataFrame({'y_true': y_test, 'y_pred': y_pred}).to_csv(os.path.join(outputs_dir, f'predictions_fold{i+1}.csv'), index=False)

        # save model
        model_path = os.path.join(models_dir, f'rotation_forest_fold{i+1}.joblib')
        save_model(model, model_path)

        all_true.append(y_test)
        all_pred.append(y_pred)

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    # overall metrics
    overall = {
        'r2_mean': float(np.mean([m['r2'] for m in fold_metrics])),
        'r2_std': float(np.std([m['r2'] for m in fold_metrics])),
        'rmse_mean': float(np.mean([m['rmse'] for m in fold_metrics])),
        'rmse_std': float(np.std([m['rmse'] for m in fold_metrics])),
        'mae_mean': float(np.mean([m['mae'] for m in fold_metrics])),
        'mae_std': float(np.std([m['mae'] for m in fold_metrics])),
    }

    # save combined predictions
    pd.DataFrame({'y_true': all_true, 'y_pred': all_pred}).to_csv(os.path.join(outputs_dir, 'predictions_cv_all.csv'), index=False)

    # parity plot
    try:
        plt.figure(figsize=(6, 6))
        plt.scatter(all_true, all_pred, s=8)
        mn = min(all_true.min(), all_pred.min())
        mx = max(all_true.max(), all_pred.max())
        plt.plot([mn, mx], [mn, mx], 'r--')
        plt.xlabel('y_true')
        plt.ylabel('y_pred')
        plt.title('CV Parity plot')
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, 'parity_plot_cv.png'))
        plt.close()
    except Exception:
        pass

    # save logs
    with open(os.path.join(logs_dir, 'summary_cv.txt'), 'w', encoding='utf8') as f:
        f.write('source: %s\n' % source_path)
        f.write('local_copy: %s\n' % local_copy_path)
        f.write('target: %s\n' % target_col)
        f.write('fold_metrics:\n')
        for m in fold_metrics:
            f.write(str(m) + '\n')
        f.write('overall:\n')
        f.write(str(overall) + '\n')

    print('CV complete. Results saved to:', out_base)
    return fold_metrics, overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=None, help='Path to source CSV to import into this folder')
    parser.add_argument('--target', default=None, help='Target column name (optional)')
    parser.add_argument('--n_estimators', type=int, default=46)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--sample_percent', type=float, default=0.75)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--random_state', type=int, default=0)
    args = parser.parse_args()

    # default source: workspace parent data/tmc_data/tmc_all_materials.csv
    if args.source is None:
        repo_root = os.path.dirname(BASE_DIR)
        default_source = os.path.join(repo_root, 'data', 'tmc_data', 'tmc_all_materials.csv')
    else:
        default_source = args.source

    if not os.path.isfile(default_source):
        raise FileNotFoundError('默认数据文件未找到: %s。请用 --source 指定路径。' % default_source)

    local_copy = os.path.join(BASE_DIR, 'data', os.path.basename(default_source))
    print('Importing source to local copy:', local_copy)
    fold_metrics, overall = run_cv(default_source, local_copy, args.target, n_estimators=args.n_estimators,
                                   K=args.K, sample_percent=args.sample_percent, folds=args.folds, random_state=args.random_state)
    print('Fold metrics:', fold_metrics)
    print('Overall:', overall)


if __name__ == '__main__':
    main()

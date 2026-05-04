import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score
import matplotlib.pyplot as plt
import joblib

# ensure local folder on path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from rotation_forest import RotationForestRegressor, RotationForestClassifier, save_model
from weka_utils import ensure_weka_available


def generate_synthetic(path, n_samples=200, n_features=8, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n_samples, n_features)
    coefs = rng.randn(n_features)
    y = X.dot(coefs) + 0.1 * rng.randn(n_samples)
    cols = ['f%d' % i for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df['target'] = y
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=os.path.join(BASE_DIR, 'data', 'sample_regression.csv'))
    parser.add_argument('--task', choices=['regression', 'classification'], default='regression')
    parser.add_argument('--target', default='target')
    parser.add_argument('--n_estimators', type=int, default=46)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--sample_percent', type=float, default=0.75)
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--use_weka', action='store_true', help='Try to use Weka REPTree (requires weka.jar and java).')
    parser.add_argument('--weka_jar', default=os.path.join(BASE_DIR, 'weka.jar'))
    args = parser.parse_args()

    data_path = args.data
    if not os.path.isfile(data_path):
        print('Data file not found, generating synthetic data at:', data_path)
        generate_synthetic(data_path)

    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        raise ValueError('目标列 %s 不存在于数据中' % args.target)

    X = df.drop(columns=[args.target]).values
    y = df[args.target].values

    outputs_dir = os.path.join(BASE_DIR, 'experiments', 'outputs')
    models_dir = os.path.join(BASE_DIR, 'experiments', 'models')
    logs_dir = os.path.join(BASE_DIR, 'experiments', 'logs')
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.random_state)

    if args.use_weka and ensure_weka_available(args.weka_jar):
        print('检测到 weka.jar，但当前脚本使用 sklearn 回退（Weka 支持在 weka_utils 中提供，但需要 Java 环境）。')
    if args.task == 'regression':
        model = RotationForestRegressor(n_estimators=args.n_estimators, K=args.K,
                                       sample_percent=args.sample_percent, random_state=args.random_state)
    else:
        model = RotationForestClassifier(n_estimators=args.n_estimators, K=args.K,
                                         sample_percent=args.sample_percent, random_state=args.random_state)

    print('Training: n_estimators=%d K=%d' % (args.n_estimators, args.K))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if args.task == 'regression':
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        summary = {'r2': r2, 'rmse': rmse, 'mae': mae}
        print('Results:', summary)
    else:
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        summary = {'accuracy': acc, 'f1_weighted': f1}
        print('Results:', summary)

    preds_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})
    preds_csv = os.path.join(outputs_dir, 'predictions.csv')
    preds_df.to_csv(preds_csv, index=False)

    model_path = os.path.join(models_dir, 'rotation_forest_model.joblib')
    save_model(model, model_path)

    try:
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, s=10)
        mn = min(y_test.min(), y_pred.min())
        mx = max(y_test.max(), y_pred.max())
        plt.plot([mn, mx], [mn, mx], 'r--')
        plt.xlabel('y_true')
        plt.ylabel('y_pred')
        plt.title('Parity plot')
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, 'parity_plot.png'))
        plt.close()
    except Exception as e:
        print('Plot failed:', e)

    # 保存简短日志
    with open(os.path.join(logs_dir, 'summary.txt'), 'w', encoding='utf8') as f:
        f.write(str(summary) + '\n')
        f.write('model: %s\n' % model_path)
        f.write('predictions: %s\n' % preds_csv)

    print('Done. All I/O confined to directory:', BASE_DIR)


if __name__ == '__main__':
    main()

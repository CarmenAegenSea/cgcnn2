import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_parity(df, out_path):
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, s=20, alpha=0.7)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2)
    ax.fill_between([mn, mx], [mn - 0.3, mx - 0.3], [mn + 0.3, mx + 0.3], color='gray', alpha=0.15)
    ax.set_xlabel('DFT (true)')
    ax.set_ylabel('Predicted')
    ax.set_title('Parity plot')
    text = f'MAE = {mae:.3f}\n$R^2$ = {r2:.3f}'
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=12, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_error_distribution(df, out_path):
    errors = df['y_pred'].values - df['y_true'].values
    mu = np.mean(errors)
    sigma = np.std(errors)

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(9, 6))
    counts, bins, patches = ax.hist(errors, bins=80, density=True, alpha=0.7, color='#4c72b0', edgecolor='k')
    x = np.linspace(errors.min(), errors.max(), 400)
    pdf = 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, pdf, 'r-', linewidth=2)
    ax.axvline(mu, color='k', linestyle='--')
    ax.set_xlabel('Prediction error')
    ax.set_ylabel('Density')
    ax.set_title('Prediction error distribution')
    ax.legend([f'Normal fit (μ={mu:.3f}, σ={sigma:.3f})'])
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_training_curve(model, X_train, y_train, X_test, y_test, out_path):
    # model is RotationForestRegressor with estimators_ and rotation_matrices_
    n_estimators = len(model.rotation_matrices_)
    train_mae = []
    test_mae = []
    for k in range(1, n_estimators + 1):
        Rs = model.rotation_matrices_[:k]
        ests = model.estimators_[:k]
        preds_train = np.mean([e.predict(X_train.dot(R)) for e, R in zip(ests, Rs)], axis=0)
        preds_test = np.mean([e.predict(X_test.dot(R)) for e, R in zip(ests, Rs)], axis=0)
        train_mae.append(float(mean_absolute_error(y_train, preds_train)))
        test_mae.append(float(mean_absolute_error(y_test, preds_test)))

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, n_estimators + 1)
    ax.plot(x, train_mae, label='Train MAE', color='blue')
    ax.plot(x, test_mae, label='Test MAE', color='red')
    ax.set_xlabel('Number of estimators')
    ax.set_ylabel('MAE')
    ax.set_title('Training progress (by number of estimators)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=os.path.join(BASE_DIR, 'data', 'catalysis.csv'))
    parser.add_argument('--target', default='formation_energy_per_atom')
    parser.add_argument('--model', default=os.path.join(BASE_DIR, 'experiments', 'models', 'rotation_forest_model.joblib'))
    parser.add_argument('--predictions', default=os.path.join(BASE_DIR, 'experiments', 'outputs', 'predictions.csv'))
    parser.add_argument('--outdir', default=os.path.join(BASE_DIR, 'experiments', 'outputs'))
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # load predictions if available
    if os.path.isfile(args.predictions):
        df_preds = pd.read_csv(args.predictions)
        if not {'y_true', 'y_pred'}.issubset(df_preds.columns):
            print('Predictions file missing required columns')
            df_preds = None
    else:
        df_preds = None

    # load model
    if not os.path.isfile(args.model):
        print('Model file not found:', args.model)
        sys.exit(1)
    model = joblib.load(args.model)

    # If predictions missing, reconstruct test set and produce predictions
    if df_preds is None:
        # read data and split
        if not os.path.isfile(args.data):
            print('Data file not found:', args.data)
            sys.exit(1)
        df = pd.read_csv(args.data)
        if args.target not in df.columns:
            raise ValueError('Target column not found in data')
        df = df.copy()
        df[args.target] = pd.to_numeric(df[args.target], errors='coerce')
        numeric = df.select_dtypes(include=[np.number])
        X = numeric.drop(columns=[args.target]).values
        y = numeric[args.target].values
        mask = ~np.isnan(y)
        mask = mask & (~np.isnan(X).any(axis=1))
        X = X[mask]
        y = y[mask]
        rng = getattr(model, 'random_state', 0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rng)
        y_pred = model.predict(X_test)
        df_preds = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})
        df_preds.to_csv(os.path.join(outdir, 'predictions.csv'), index=False)
    else:
        # also reconstruct train/test split for curve
        if not os.path.isfile(args.data):
            print('Data file not found for training curve, skipping training curve')
            X_train = X_test = y_train = y_test = None
        else:
            df = pd.read_csv(args.data)
            df = df.copy()
            df[args.target] = pd.to_numeric(df[args.target], errors='coerce')
            numeric = df.select_dtypes(include=[np.number])
            X = numeric.drop(columns=[args.target]).values
            y = numeric[args.target].values
            mask = ~np.isnan(y)
            mask = mask & (~np.isnan(X).any(axis=1))
            X = X[mask]
            y = y[mask]
            rng = getattr(model, 'random_state', 0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rng)

    # parity
    print('Plotting parity...')
    plot_parity(df_preds, os.path.join(outdir, 'parity_plot_custom.png'))

    # error distribution
    print('Plotting error distribution...')
    plot_error_distribution(df_preds, os.path.join(outdir, 'error_dist.png'))

    # training curve
    if X_train is not None and X_test is not None and hasattr(model, 'estimators_') and hasattr(model, 'rotation_matrices_'):
        print('Plotting training curve...')
        plot_training_curve(model, X_train, y_train, X_test, y_test, os.path.join(outdir, 'training_curve_estimators.png'))
    else:
        print('Skipping training curve (missing data or model internals)')

    print('Plots saved to', outdir)


if __name__ == '__main__':
    main()

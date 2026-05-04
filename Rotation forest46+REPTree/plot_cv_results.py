import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample, check_random_state
from sklearn.model_selection import KFold


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_combined_predictions(outputs_dir):
    f_all = os.path.join(outputs_dir, 'predictions_cv_all.csv')
    if os.path.isfile(f_all):
        df = pd.read_csv(f_all)
        if {'y_true', 'y_pred'}.issubset(df.columns):
            return df
    # fallback: combine per-fold files
    rows = []
    for name in sorted(os.listdir(outputs_dir)):
        if name.startswith('predictions_fold') and name.endswith('.csv'):
            df2 = pd.read_csv(os.path.join(outputs_dir, name))
            if {'y_true', 'y_pred'}.issubset(df2.columns):
                rows.append(df2[['y_true', 'y_pred']])
    if rows:
        return pd.concat(rows, ignore_index=True)
    return None


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
    # ±0.3 band
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


def incremental_training_curve(X, y, n_estimators=46, K=3, sample_percent=0.75, random_state=0):
    n_samples, n_features = X.shape
    rng_base = check_random_state(random_state)
    estimators = []
    rotations = []
    train_mae = []
    # We'll return training MAE per estimator when called with training and validation sets
    for i in range(n_estimators):
        rng = check_random_state(rng_base.randint(0, 2 ** 31 - 1))
        feat_idx = rng.permutation(n_features)
        subsets = np.array_split(feat_idx, K)
        R = np.zeros((n_features, n_features))
        for subset in subsets:
            if len(subset) == 0:
                continue
            n_sel = max(len(subset), int(max(1, X.shape[0] * sample_percent)))
            sel = resample(np.arange(X.shape[0]), replace=True, n_samples=n_sel, random_state=rng.randint(0, 2 ** 31 - 1))
            Xsub = X[sel][:, subset]
            try:
                pca = PCA(n_components=len(subset), svd_solver='full', random_state=rng.randint(0, 2 ** 31 - 1))
                pca.fit(Xsub)
                comp = pca.components_.T
                if comp.shape[0] != len(subset) or comp.shape[1] != len(subset):
                    comp = np.eye(len(subset))
            except Exception:
                comp = np.eye(len(subset))
            R[np.ix_(subset, subset)] = comp

        # fit estimator on whole X rotated
        Xrot = X.dot(R)
        est = DecisionTreeRegressor(random_state=rng.randint(0, 2 ** 31 - 1))
        est.fit(Xrot, y)
        estimators.append(est)
        rotations.append(R)
        # compute ensemble prediction on X (training)
        preds = np.array([e.predict(X.dot(Rj)) for e, Rj in zip(estimators, rotations)])
        preds_mean = preds.mean(axis=0)
        train_mae.append(float(mean_absolute_error(y, preds_mean)))
    return train_mae


def plot_training_curve_from_cv(source_csv, target_col, n_estimators=46, K=3, sample_percent=0.75, folds=5, random_state=0, out_path=None):
    df = pd.read_csv(source_csv)
    if target_col is None:
        # try to choose
        if 'formation_energy_per_atom' in df.columns:
            target_col = 'formation_energy_per_atom'
        else:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) == 0:
                raise ValueError('No numeric columns found to use as target')
            target_col = num_cols[-1]

    numeric = df.select_dtypes(include=[np.number])
    if target_col not in numeric.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        numeric = df.select_dtypes(include=[np.number])
    X_all = numeric.drop(columns=[target_col]).values
    y_all = numeric[target_col].values
    mask = ~np.isnan(y_all)
    mask = mask & (~np.isnan(X_all).any(axis=1))
    X_all = X_all[mask]
    y_all = y_all[mask]

    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    train_seqs = []
    val_seqs = []
    for i, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        X_train, X_val = X_all[train_idx], X_all[test_idx]
        y_train, y_val = y_all[train_idx], y_all[test_idx]

        # build estimators sequentially and record train/val MAE per estimator
        estimators = []
        rotations = []
        train_mae = []
        val_mae = []
        n_features = X_train.shape[1]
        rng_base = check_random_state(random_state + i)
        for j in range(n_estimators):
            rng = check_random_state(rng_base.randint(0, 2 ** 31 - 1))
            feat_idx = rng.permutation(n_features)
            subsets = np.array_split(feat_idx, K)
            R = np.zeros((n_features, n_features))
            for subset in subsets:
                if len(subset) == 0:
                    continue
                n_sel = max(len(subset), int(max(1, X_train.shape[0] * sample_percent)))
                sel = resample(np.arange(X_train.shape[0]), replace=True, n_samples=n_sel, random_state=rng.randint(0, 2 ** 31 - 1))
                Xsub = X_train[sel][:, subset]
                try:
                    pca = PCA(n_components=len(subset), svd_solver='full', random_state=rng.randint(0, 2 ** 31 - 1))
                    pca.fit(Xsub)
                    comp = pca.components_.T
                    if comp.shape[0] != len(subset) or comp.shape[1] != len(subset):
                        comp = np.eye(len(subset))
                except Exception:
                    comp = np.eye(len(subset))
                R[np.ix_(subset, subset)] = comp

            Xtr_rot = X_train.dot(R)
            est = DecisionTreeRegressor(random_state=rng.randint(0, 2 ** 31 - 1))
            est.fit(Xtr_rot, y_train)
            estimators.append(est)
            rotations.append(R)

            preds_tr = np.mean([e.predict(X_train.dot(Rj)) for e, Rj in zip(estimators, rotations)], axis=0)
            preds_val = np.mean([e.predict(X_val.dot(Rj)) for e, Rj in zip(estimators, rotations)], axis=0)
            train_mae.append(float(mean_absolute_error(y_train, preds_tr)))
            val_mae.append(float(mean_absolute_error(y_val, preds_val)))

        train_seqs.append(train_mae)
        val_seqs.append(val_mae)

    train_mean = np.mean(train_seqs, axis=0)
    train_std = np.std(train_seqs, axis=0)
    val_mean = np.mean(val_seqs, axis=0)
    val_std = np.std(val_seqs, axis=0)

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, n_estimators + 1)
    ax.plot(x, train_mean, label='Train MAE', color='blue')
    ax.fill_between(x, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
    ax.plot(x, val_mean, label='Validation MAE', color='red')
    ax.fill_between(x, val_mean - val_std, val_mean + val_std, color='red', alpha=0.2)
    ax.set_xlabel('Number of estimators')
    ax.set_ylabel('MAE')
    ax.set_title('Training progress (by number of estimators)')
    ax.legend()
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200)
    else:
        fig.savefig(os.path.join(BASE_DIR, 'experiments', 'cv', 'outputs', 'training_curve_estimators.png'), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=os.path.join(BASE_DIR, 'data', 'catalysis.csv'))
    parser.add_argument('--target', default='formation_energy_per_atom')
    parser.add_argument('--n_estimators', type=int, default=46)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--sample_percent', type=float, default=0.75)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--random_state', type=int, default=0)
    args = parser.parse_args()

    outputs_dir = os.path.join(BASE_DIR, 'experiments', 'cv', 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)

    # load combined predictions for parity and error dist
    df_preds = load_combined_predictions(outputs_dir)
    if df_preds is None:
        print('Combined predictions not found in', outputs_dir)
        sys.exit(1)

    print('Plotting parity...')
    plot_parity(df_preds, os.path.join(outputs_dir, 'parity_plot_custom.png'))
    print('Plotting error distribution...')
    plot_error_distribution(df_preds, os.path.join(outputs_dir, 'error_dist.png'))

    print('Computing training curve (this may take a little time)...')
    plot_training_curve_from_cv(args.source, args.target, n_estimators=args.n_estimators, K=args.K,
                                sample_percent=args.sample_percent, folds=args.folds, random_state=args.random_state,
                                out_path=os.path.join(outputs_dir, 'training_curve_estimators.png'))

    print('Plots saved to', outputs_dir)


if __name__ == '__main__':
    main()

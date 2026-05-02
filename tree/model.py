import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None,
                        random_state=42, n_jobs=-1):
    """Train a RandomForestRegressor with standard scaling on features.

    Returns (scaler, model)
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    rf = RandomForestRegressor(n_estimators=n_estimators,
                               max_depth=max_depth,
                               random_state=random_state,
                               n_jobs=n_jobs)
    rf.fit(Xs, y_train)
    return scaler, rf


def evaluate_model(scaler, model, X, y_true):
    Xs = scaler.transform(X)
    # predictions from whole forest
    preds = model.predict(Xs)
    mae = float(mean_absolute_error(y_true, preds))
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    return {"mae": mae, "rmse": rmse}


def predict_with_uncertainty(scaler, model, X):
    """Return mean and std of per-tree predictions.

    Output: preds_mean (n,), preds_std (n,)
    """
    Xs = scaler.transform(X)
    # collect per-tree predictions
    all_preds = np.vstack([est.predict(Xs) for est in model.estimators_])
    mean_preds = np.mean(all_preds, axis=0)
    std_preds = np.std(all_preds, axis=0)
    return mean_preds, std_preds


def save_model_bundle(out_path, scaler, model, element_list):
    bundle = {
        'model': model,
        'scaler': scaler,
        'elements': list(element_list),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(bundle, out_path)


def load_model_bundle(path):
    return joblib.load(path)


if __name__ == '__main__':
    print('This module provides helpers to train / evaluate a RandomForest baseline.')

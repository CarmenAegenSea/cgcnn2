import os
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.decomposition import PCA
from sklearn.utils import resample, check_random_state
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import joblib


def _majority_vote(preds):
    preds = np.asarray(preds)
    if preds.ndim == 1:
        return preds
    n_samples = preds.shape[1]
    out = np.empty(n_samples, dtype=preds.dtype)
    for i in range(n_samples):
        vals, counts = np.unique(preds[:, i], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


class RotationForestRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator=None, n_estimators=46, K=3, sample_percent=0.75, random_state=None):
        self.base_estimator = base_estimator if base_estimator is not None else DecisionTreeRegressor()
        self.n_estimators = int(n_estimators)
        self.K = int(K)
        self.sample_percent = float(sample_percent)
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        rng = check_random_state(self.random_state)
        n_features = X.shape[1]
        self.estimators_ = []
        self.rotation_matrices_ = []
        for i in range(self.n_estimators):
            feat_idx = rng.permutation(n_features)
            subsets = np.array_split(feat_idx, self.K)
            R = np.zeros((n_features, n_features))
            for subset in subsets:
                if len(subset) == 0:
                    continue
                n_sel = max(len(subset), int(max(1, X.shape[0] * self.sample_percent)))
                sel = resample(np.arange(X.shape[0]), replace=True, n_samples=n_sel,
                               random_state=rng.randint(0, 2 ** 31 - 1))
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
            Xrot = X.dot(R)
            est = clone(self.base_estimator)
            est.fit(Xrot, y)
            self.estimators_.append(est)
            self.rotation_matrices_.append(R)
        return self

    def predict(self, X):
        X = np.asarray(X)
        preds = np.array([est.predict(X.dot(R)) for est, R in zip(self.estimators_, self.rotation_matrices_)])
        return preds.mean(axis=0)


class RotationForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, n_estimators=46, K=3, sample_percent=0.75, random_state=None):
        self.base_estimator = base_estimator if base_estimator is not None else DecisionTreeClassifier()
        self.n_estimators = int(n_estimators)
        self.K = int(K)
        self.sample_percent = float(sample_percent)
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        rng = check_random_state(self.random_state)
        n_features = X.shape[1]
        self.estimators_ = []
        self.rotation_matrices_ = []
        for i in range(self.n_estimators):
            feat_idx = rng.permutation(n_features)
            subsets = np.array_split(feat_idx, self.K)
            R = np.zeros((n_features, n_features))
            for subset in subsets:
                if len(subset) == 0:
                    continue
                n_sel = max(len(subset), int(max(1, X.shape[0] * self.sample_percent)))
                sel = resample(np.arange(X.shape[0]), replace=True, n_samples=n_sel,
                               random_state=rng.randint(0, 2 ** 31 - 1))
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
            Xrot = X.dot(R)
            est = clone(self.base_estimator)
            est.fit(Xrot, y)
            self.estimators_.append(est)
            self.rotation_matrices_.append(R)
        return self

    def predict(self, X):
        X = np.asarray(X)
        preds = np.array([est.predict(X.dot(R)) for est, R in zip(self.estimators_, self.rotation_matrices_)])
        return _majority_vote(preds)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


__all__ = [
    'RotationForestRegressor',
    'RotationForestClassifier',
    'save_model',
    'load_model',
]

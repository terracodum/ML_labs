"""
Пользовательская реализация классификатора kNN в стиле scikit-learn.
Подключается к ноутбуку: from custom_knn import CustomKNNClassifier
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class CustomKNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, metric="euclidean"):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)
        self.classes_ = np.unique(self.y_train_)
        return self

    def _compute_distances(self, X):
        X = np.asarray(X, dtype=float)
        if self.metric == "euclidean":
            diff = self.X_train_[None, :, :] - X[:, None, :]
            return np.sqrt(np.sum(diff ** 2, axis=2))
        if self.metric == "manhattan":
            diff = np.abs(self.X_train_[None, :, :] - X[:, None, :])
            return np.sum(diff, axis=2)
        raise ValueError("metric must be 'euclidean' or 'manhattan'")

    def predict_proba(self, X):
        distances = self._compute_distances(X)
        neighbor_idx = np.argsort(distances, axis=1)[:, : self.n_neighbors]
        neighbor_labels = self.y_train_[neighbor_idx]

        proba = np.zeros((neighbor_labels.shape[0], len(self.classes_)), dtype=float)
        for row_idx, row_labels in enumerate(neighbor_labels):
            for class_idx, class_label in enumerate(self.classes_):
                proba[row_idx, class_idx] = np.mean(row_labels == class_label)
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

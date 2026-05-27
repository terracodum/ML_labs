"""
Библиотека алгоритмов ML — Лабораторная работа 5.
Собственная реализация алгоритма кластеризации K-Means.
"""

import numpy as np


class CustomKMeans:
    """
    Реализация K-Means с подсчётом WCSS (Within-Cluster Sum of Squares).

    Parameters
    ----------
    n_clusters : int
        Количество кластеров.
    max_iter : int
        Максимальное число итераций.
    tol : float
        Порог сходимости (максимальный сдвиг центроиды).
    random_state : int or None
        Seed для воспроизводимости.
    """

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X):
        """Обучить модель на данных X."""
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        # Случайная инициализация центроид из точек данных
        idx = rng.choice(len(X), self.n_clusters, replace=False)
        self.cluster_centers_ = X[idx].copy()

        for iteration in range(self.max_iter):
            labels = self._assign(X)

            # M-шаг: обновить центроиды как среднее кластера
            new_centers = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k)
                else self.cluster_centers_[k]          # оставить если кластер пуст
                for k in range(self.n_clusters)
            ])

            # Проверка сходимости по максимальному сдвигу
            shift = np.max(np.linalg.norm(new_centers - self.cluster_centers_, axis=1))
            self.cluster_centers_ = new_centers

            if shift < self.tol:
                break

        self.labels_ = self._assign(X)
        self.inertia_ = self._wcss(X)
        self.n_iter_ = iteration + 1
        return self

    def predict(self, X):
        """Предсказать метки кластеров для X."""
        X = np.asarray(X, dtype=float)
        return self._assign(X)

    def fit_predict(self, X):
        """Обучить и вернуть метки кластеров."""
        return self.fit(X).labels_

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assign(self, X):
        """E-шаг: присвоить каждой точке ближайшую центроиду."""
        dists = np.linalg.norm(
            X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :],
            axis=2
        )
        return np.argmin(dists, axis=1)

    def _wcss(self, X):
        """
        WCSS — сумма квадратов расстояний от точек до центроид своего кластера.
        Аналог sklearn KMeans.inertia_.
        """
        labels = self._assign(X)
        return float(sum(
            np.sum((X[labels == k] - self.cluster_centers_[k]) ** 2)
            for k in range(self.n_clusters)
            if np.any(labels == k)
        ))

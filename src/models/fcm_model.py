"""Fuzzy C-Means (FCM) Clustering Implementation (Bezdek, 1984).

Provides a vectorized NumPy implementation of Bezdek's Fuzzy C-Means algorithm
with membership matrix tracking, fuzzifier weighting exponent (m), objective
progression J_m, Fuzzy Partition Coefficient (FPC), and crisp defuzzification.
"""

import time
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from scipy.spatial.distance import cdist
from .base import BaseClusterer


class FuzzyCMeansModel(BaseClusterer):
    """Fuzzy C-Means (Soft/Fuzzy) Clustering Model."""

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        super().__init__(n_clusters=n_clusters, max_iter=max_iter, tol=tol, random_state=random_state)
        if m <= 1.0:
            raise ValueError(f"Fuzzifier exponent 'm' must be strictly greater than 1.0, got {m}")
        self.m = float(m)

        self.u_: Optional[np.ndarray] = None  # Membership matrix (n_samples, n_clusters)
        self.fpc_: float = 0.0               # Fuzzy Partition Coefficient
        self.crisp_sse_: float = 0.0         # Sum of Squared Errors with defuzzified labels
        self.fuzzy_sse_: float = 0.0         # Fuzzy weighted objective J_m

    def _init_membership(self, n_samples: int) -> np.ndarray:
        """Initialize random membership matrix where each row sums to 1.0."""
        rng = np.random.RandomState(self.random_state)
        u = rng.uniform(0.0, 1.0, size=(n_samples, self.n_clusters))
        u = u / np.sum(u, axis=1, keepdims=True)
        return u

    def _compute_centers(self, X: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute cluster centroids from membership matrix: v_j = sum(u_ij^m * x_i) / sum(u_ij^m)."""
        um = u ** self.m
        denominator = np.sum(um, axis=0, keepdims=True).T  # (n_clusters, 1)
        # Avoid division by zero
        denominator = np.maximum(denominator, 1e-12)
        centers = (um.T @ X) / denominator
        return centers

    def _compute_membership(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Update membership matrix based on distances to cluster centers."""
        # distances shape: (n_samples, n_clusters)
        dists = cdist(X, centers, metric="euclidean")
        # Clamp zero distance to small epsilon
        dists = np.maximum(dists, 1e-12)

        power = 2.0 / (self.m - 1.0)
        # u_ij = 1 / sum_l ( (d_ij / d_il) ** (2 / (m - 1)) )
        # Vectorized: dists[:, :, None] / dists[:, None, :] -> shape (n_samples, n_clusters, n_clusters)
        # Or simpler: inv_dists = (1.0 / dists) ** power -> normalize across clusters
        inv_dists = (1.0 / dists) ** power
        u_new = inv_dists / np.sum(inv_dists, axis=1, keepdims=True)
        return u_new

    def _compute_objective(self, X: np.ndarray, centers: np.ndarray, u: np.ndarray) -> float:
        """Compute fuzzy objective function J_m = sum(u_ij^m * ||x_i - v_j||^2)."""
        dists_sq = cdist(X, centers, metric="sqeuclidean")
        um = u ** self.m
        return float(np.sum(um * dists_sq))

    def fit(self, X: np.ndarray) -> "FuzzyCMeansModel":
        """Fit Fuzzy C-Means to feature matrix X."""
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        start_time = time.perf_counter()

        u = self._init_membership(n_samples)
        self.objective_history_ = []

        for iteration in range(1, self.max_iter + 1):
            centers = self._compute_centers(X, u)
            u_new = self._compute_membership(X, centers)
            obj = self._compute_objective(X, centers, u_new)
            self.objective_history_.append(obj)

            # Check convergence: Frobenius norm difference of membership matrix
            u_diff = np.linalg.norm(u_new - u)
            u = u_new

            if u_diff < self.tol:
                self.n_iter_ = iteration
                break
        else:
            self.n_iter_ = self.max_iter

        self.execution_time_ms_ = (time.perf_counter() - start_time) * 1000.0
        self.u_ = u
        self.cluster_centers_ = self._compute_centers(X, u)
        self.labels_ = np.argmax(u, axis=1)

        # Fuzzy Partition Coefficient: FPC = sum(u_ij^2) / n
        self.fpc_ = float(np.mean(np.sum(u ** 2, axis=1)))
        self.fuzzy_sse_ = float(self.objective_history_[-1])

        # Crisp SSE calculation
        closest_centers = self.cluster_centers_[self.labels_]
        self.crisp_sse_ = float(np.sum((X - closest_centers) ** 2))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict crisp cluster labels for new samples in X."""
        u = self.predict_proba(X)
        return np.argmax(u, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return fuzzy membership degrees for samples in X."""
        if self.cluster_centers_ is None:
            raise ValueError("Model is not fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        return self._compute_membership(X, self.cluster_centers_)

    def get_membership_matrix(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Return membership matrix u (n_samples, n_clusters)."""
        if X is None:
            if self.u_ is None:
                raise ValueError("Model is not fitted yet.")
            return self.u_
        return self.predict_proba(X)

    def get_diagnostics(self) -> Dict[str, Any]:
        diag = super().get_diagnostics()
        diag["fpc"] = round(self.fpc_, 4)
        diag["fuzzy_sse"] = round(self.fuzzy_sse_, 4)
        diag["crisp_sse"] = round(self.crisp_sse_, 4)
        diag["cluster_sizes"] = np.bincount(self.labels_, minlength=self.n_clusters).tolist()
        return diag

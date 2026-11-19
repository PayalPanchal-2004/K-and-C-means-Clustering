"""K-Means Clustering Implementation with Diagnostics and Trajectory Tracking."""

import time
from typing import Optional, List, Dict, Any
import numpy as np
from sklearn.cluster import KMeans
from .base import BaseClusterer


class KMeansModel(BaseClusterer):
    """K-Means (Hard/Crisp) Clustering Model."""

    def __init__(
        self,
        n_clusters: int = 3,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        super().__init__(n_clusters=n_clusters, max_iter=max_iter, tol=tol, random_state=random_state)
        self.init = init
        self.n_init = n_init
        self.inertia_: float = 0.0
        self._sklearn_model: Optional[KMeans] = None

    def fit(self, X: np.ndarray) -> "KMeansModel":
        """Fit K-Means model to X and record convergence diagnostics."""
        X = np.asarray(X, dtype=np.float64)
        start_time = time.perf_counter()

        self._sklearn_model = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        self._sklearn_model.fit(X)

        self.execution_time_ms_ = (time.perf_counter() - start_time) * 1000.0
        self.labels_ = self._sklearn_model.labels_
        self.cluster_centers_ = self._sklearn_model.cluster_centers_
        self.inertia_ = float(self._sklearn_model.inertia_)
        self.n_iter_ = int(self._sklearn_model.n_iter_)

        # In K-Means, objective function is SSE (inertia)
        self.objective_history_ = [self.inertia_]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict closest cluster for each sample in X."""
        if self._sklearn_model is None:
            raise ValueError("Model is not fitted yet.")
        return self._sklearn_model.predict(np.asarray(X, dtype=np.float64))

    def get_membership_matrix(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Return hard binary one-hot membership matrix (n_samples, n_clusters) for symmetry with FCM."""
        labels = self.labels_ if X is None else self.predict(X)
        n_samples = len(labels)
        u_matrix = np.zeros((n_samples, self.n_clusters), dtype=np.float64)
        u_matrix[np.arange(n_samples), labels] = 1.0
        return u_matrix

    def get_diagnostics(self) -> Dict[str, Any]:
        diag = super().get_diagnostics()
        diag["inertia_sse"] = round(self.inertia_, 4)
        diag["cluster_sizes"] = np.bincount(self.labels_, minlength=self.n_clusters).tolist()
        return diag

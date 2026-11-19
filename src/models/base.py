"""Abstract Base Class for Clustering Models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BaseClusterer(ABC):
    """Abstract Base Class ensuring API symmetry between K-Means and Fuzzy C-Means."""

    def __init__(self, n_clusters: int = 3, max_iter: int = 300, tol: float = 1e-4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.n_iter_: int = 0
        self.execution_time_ms_: float = 0.0
        self.objective_history_: list = []

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseClusterer":
        """Fit clusterer to feature matrix X."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict crisp cluster labels for samples in X."""
        pass

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit model and return crisp cluster labels."""
        self.fit(X)
        return self.labels_

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic metrics of the fitted model."""
        return {
            "n_clusters": self.n_clusters,
            "n_iter": self.n_iter_,
            "execution_time_ms": round(self.execution_time_ms_, 2),
            "final_objective": self.objective_history_[-1] if self.objective_history_ else None,
            "objective_history": self.objective_history_,
        }

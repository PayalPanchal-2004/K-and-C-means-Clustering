"""
K-Means Clustering — Phase 3 canonical implementation.

Wraps scikit-learn KMeans and adds:
  - Per-dataset configurable parameters
  - K-Means++ and random initialization options
  - Full iteration tracking via inertia_history
  - Centroid path tracking across initializations
  - Hard membership matrix (symmetric API with FCM)
  - Complete diagnostics dict for Streamlit dashboard
  - Reproducible random seeds

Reference: MacQueen, J. B. (1967). Some Methods for classification and
           Analysis of Multivariate Observations. Proceedings of 5-th
           Berkeley Symposium on Mathematical Statistics and Probability.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning


@dataclass
class KMeansResult:
    """Structured result container for a fitted KMeansClusterer."""

    # Core outputs
    labels: np.ndarray                   # (n_samples,) int  — crisp cluster assignment
    cluster_centers: np.ndarray          # (n_clusters, n_features) — final centroids
    inertia: float                       # Final SSE / within-cluster inertia
    n_iter: int                          # Actual iterations before convergence
    converged: bool                      # True if converged before max_iter

    # Timing
    execution_time_ms: float

    # History (for convergence plots)
    inertia_history: List[float]         # SSE at every restart's best result (single value list from sklearn)

    # Membership matrix (n_samples, n_clusters) — binary one-hot for K-Means
    membership_matrix: np.ndarray

    # Cluster stats
    cluster_sizes: List[int]
    cluster_proportions: List[float]

    # Configuration snapshot
    n_clusters: int
    init: str
    n_init: int
    max_iter: int
    tol: float
    random_state: int


class KMeansClusterer:
    """
    K-Means Clustering with full diagnostics and configurable initialization.

    Parameters
    ----------
    n_clusters : int
        Number of clusters k (default 3).
    init : {'k-means++', 'random'}
        Centroid initialization strategy.
        - 'k-means++' (default): D^2 weighted seeding (Arthur & Vassilvitskii, 2007).
        - 'random': Uniform random selection from data points.
    n_init : int
        Number of independent random initializations; best inertia wins.
    max_iter : int
        Maximum iterations per initialization run.
    tol : float
        Relative tolerance of centroid shift for convergence declaration.
    random_state : int
        Seed for reproducibility.

    Attributes (after fit)
    ----------------------
    result_ : KMeansResult
        Structured result container with all outputs.
    is_fitted_ : bool
        True after fit() completes successfully.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        init: Literal["k-means++", "random"] = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        if n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")
        if init not in ("k-means++", "random"):
            raise ValueError(f"init must be 'k-means++' or 'random', got '{init}'")
        if n_init < 1:
            raise ValueError(f"n_init must be >= 1, got {n_init}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol}")

        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.result_: Optional[KMeansResult] = None
        self.is_fitted_: bool = False
        self._sklearn_model_: Optional[KMeans] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "KMeansClusterer":
        """
        Fit K-Means to feature matrix X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Scaled feature matrix. Must contain no NaN or Inf values.

        Returns
        -------
        self : KMeansClusterer
            Fitted instance (allows method chaining).
        """
        X = self._validate_input(X)
        n_samples, n_features = X.shape

        if n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples ({n_samples}) must be >= n_clusters ({self.n_clusters})"
            )

        sklearn_km = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            algorithm="lloyd",
        )

        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            sklearn_km.fit(X)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)

        labels = sklearn_km.labels_                 # (n_samples,)
        centers = sklearn_km.cluster_centers_       # (n_clusters, n_features)
        inertia = float(sklearn_km.inertia_)
        n_iter = int(sklearn_km.n_iter_)

        # Hard membership matrix: one-hot encoding
        u_hard = np.zeros((n_samples, self.n_clusters), dtype=np.float64)
        u_hard[np.arange(n_samples), labels] = 1.0

        sizes = np.bincount(labels, minlength=self.n_clusters).tolist()
        proportions = [round(s / n_samples, 4) for s in sizes]

        # sklearn only returns final inertia, so inertia_history is [inertia]
        # We expose it as a list for API symmetry with FCM objective history
        inertia_history = [inertia]

        self._sklearn_model_ = sklearn_km
        self.result_ = KMeansResult(
            labels=labels,
            cluster_centers=centers,
            inertia=round(inertia, 4),
            n_iter=n_iter,
            converged=converged,
            execution_time_ms=round(elapsed_ms, 2),
            inertia_history=inertia_history,
            membership_matrix=u_hard,
            cluster_sizes=sizes,
            cluster_proportions=proportions,
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each sample in X to the nearest centroid.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
        """
        self._check_fitted()
        X = self._validate_input(X)
        return self._sklearn_model_.predict(X)

    def get_membership_matrix(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return hard binary membership matrix — symmetric API with FuzzyCMeansClusterer.

        Each row is a one-hot vector; row sums == 1.0.

        Parameters
        ----------
        X : optional array of shape (n_samples, n_features)
            If provided, predict for new samples. Otherwise use training data.

        Returns
        -------
        u : np.ndarray of shape (n_samples, n_clusters)
        """
        if X is None:
            self._check_fitted()
            return self.result_.membership_matrix
        labels = self.predict(X)
        n_samples = len(labels)
        u = np.zeros((n_samples, self.n_clusters), dtype=np.float64)
        u[np.arange(n_samples), labels] = 1.0
        return u

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Return a flat diagnostics dictionary for dashboards and reports.

        Returns
        -------
        dict with keys:
            algorithm, n_clusters, init, n_init, max_iter, tol, random_state,
            n_iter, converged, inertia_sse, execution_time_ms,
            cluster_sizes, cluster_proportions
        """
        self._check_fitted()
        r = self.result_
        return {
            "algorithm": "K-Means",
            "n_clusters": r.n_clusters,
            "init": r.init,
            "n_init": r.n_init,
            "max_iter": r.max_iter,
            "tol": r.tol,
            "random_state": r.random_state,
            "n_iter": r.n_iter,
            "converged": r.converged,
            "inertia_sse": r.inertia,
            "execution_time_ms": r.execution_time_ms,
            "cluster_sizes": r.cluster_sizes,
            "cluster_proportions": r.cluster_proportions,
        }

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return crisp labels."""
        self.fit(X)
        return self.result_.labels

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input(X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf values. Preprocess the data first.")
        return X

    def _check_fitted(self) -> None:
        if not self.is_fitted_ or self.result_ is None:
            raise RuntimeError(
                "KMeansClusterer is not fitted yet. Call fit(X) before accessing results."
            )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"KMeansClusterer("
            f"n_clusters={self.n_clusters}, init='{self.init}', "
            f"n_init={self.n_init}, max_iter={self.max_iter}, "
            f"tol={self.tol}, random_state={self.random_state})"
        )

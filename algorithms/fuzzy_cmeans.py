"""
Fuzzy C-Means (FCM) Clustering — Phase 3 canonical implementation.

Implements the original Bezdek (1984) alternating optimization algorithm:
  1. Initialize membership matrix U with random rows summing to 1.
  2. Compute fuzzy centroids V from U and feature matrix X.
  3. Update U from distances to V.
  4. Repeat until ||U_new - U_old||_F < tol or max_iter reached.

Key properties enforced:
  - Membership constraint: for each sample i, sum_j(u_ij) == 1  (partition unity)
  - Fuzzifier: m > 1  (m=1 → crisp K-Means; m→∞ → equal membership)
  - Degenerate distances handled via epsilon clamping
  - Hard labels derived via argmax(U, axis=1) only for external metrics

Reference: Bezdek, J. C. (1984). FCM: The fuzzy c-means clustering algorithm.
           Computers & Geosciences, 10(2–3), 191–203.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FCMResult:
    """Structured result container for a fitted FuzzyCMeansClusterer."""

    # Core outputs
    labels: np.ndarray              # (n_samples,)  — argmax defuzzified labels
    cluster_centers: np.ndarray     # (n_clusters, n_features) — fuzzy centroids
    membership_matrix: np.ndarray   # (n_samples, n_clusters) — soft memberships

    # Fuzzy-specific metrics
    fpc: float                      # Fuzzy Partition Coefficient ∈ (1/n_clusters, 1]
    fpe: float                      # Fuzzy Partition Entropy ∈ [0, log(n_clusters)]
    fuzzy_sse: float                # Fuzzy objective J_m at convergence
    crisp_sse: float                # Hard SSE using defuzzified labels

    # Convergence
    n_iter: int
    converged: bool
    objective_history: List[float]  # J_m at each iteration

    # Timing
    execution_time_ms: float

    # Cluster stats
    cluster_sizes: List[int]
    cluster_proportions: List[float]

    # Configuration snapshot
    n_clusters: int
    m: float
    max_iter: int
    tol: float
    random_state: int


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FuzzyCMeansClusterer:
    """
    Fuzzy C-Means (FCM) Clustering — Bezdek (1984).

    Parameters
    ----------
    n_clusters : int
        Number of fuzzy clusters c (default 3).
    m : float
        Fuzzifier (weighting exponent). Must be > 1.0.
        - m = 2.0 : standard default.
        - m close to 1 : approaches crisp K-Means.
        - m large (> 3) : very soft, all memberships approach 1/c.
    max_iter : int
        Maximum iterations for the alternating optimization loop.
    tol : float
        Convergence threshold on ||U_new - U_old||_Frobenius.
    random_state : int
        Seed for initial membership matrix generation.

    Attributes (after fit)
    ----------------------
    result_ : FCMResult
        Structured result container.
    is_fitted_ : bool
        True after fit() completes.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        if n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")
        if m <= 1.0:
            raise ValueError(
                f"Fuzzifier 'm' must be strictly > 1.0, got {m}. "
                f"(m=1 is crisp; use KMeansClusterer instead)"
            )
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol}")

        self.n_clusters = n_clusters
        self.m = float(m)
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.result_: Optional[FCMResult] = None
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "FuzzyCMeansClusterer":
        """
        Run FCM alternating optimization on feature matrix X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Scaled feature matrix. No NaN or Inf values allowed.

        Returns
        -------
        self : FuzzyCMeansClusterer
            Fitted instance.
        """
        X = self._validate_input(X)
        n_samples, n_features = X.shape

        if n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples ({n_samples}) must be >= n_clusters ({self.n_clusters})"
            )

        start = time.perf_counter()

        # Step 1: Initialize random U — each row sums to 1
        u = self._init_membership(n_samples)
        objective_history: List[float] = []
        converged = False

        for iteration in range(1, self.max_iter + 1):
            # Step 2: Compute cluster centres V
            centers = self._compute_centers(X, u)

            # Step 3: Update membership matrix U
            u_new = self._compute_membership(X, centers)

            # Step 4: Compute fuzzy objective J_m
            obj = self._compute_objective(X, centers, u_new)
            objective_history.append(round(obj, 6))

            # Step 5: Check convergence
            delta = float(np.linalg.norm(u_new - u, ord="fro"))
            u = u_new

            if delta < self.tol:
                converged = True
                n_iter = iteration
                break
        else:
            n_iter = self.max_iter

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Final centres from converged U
        final_centers = self._compute_centers(X, u)
        labels = np.argmax(u, axis=1)

        # Fuzzy Partition Coefficient:  FPC = (1/N) * sum(u_ij^2)
        fpc = float(np.sum(u ** 2) / n_samples)

        # Fuzzy Partition Entropy:  FPE = -(1/N) * sum(u_ij * log(u_ij))
        # Clamp u to avoid log(0)
        u_safe = np.maximum(u, 1e-15)
        fpe = float(-np.sum(u_safe * np.log(u_safe)) / n_samples)

        fuzzy_sse = objective_history[-1] if objective_history else 0.0

        # Crisp SSE using defuzzified labels
        assigned = final_centers[labels]
        crisp_sse = float(np.sum((X - assigned) ** 2))

        sizes = np.bincount(labels, minlength=self.n_clusters).tolist()
        proportions = [round(s / n_samples, 4) for s in sizes]

        self.result_ = FCMResult(
            labels=labels,
            cluster_centers=final_centers,
            membership_matrix=u,
            fpc=round(fpc, 6),
            fpe=round(fpe, 6),
            fuzzy_sse=round(fuzzy_sse, 4),
            crisp_sse=round(crisp_sse, 4),
            n_iter=n_iter,
            converged=converged,
            objective_history=objective_history,
            execution_time_ms=round(elapsed_ms, 2),
            cluster_sizes=sizes,
            cluster_proportions=proportions,
            n_clusters=self.n_clusters,
            m=self.m,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign hard (crisp) cluster labels to new samples.

        Computes membership degrees and returns argmax labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
        """
        u = self.predict_proba(X)
        return np.argmax(u, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return fuzzy membership degrees for samples in X.

        Each row sums to approximately 1.0 (partition unity).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        u : np.ndarray of shape (n_samples, n_clusters)
        """
        self._check_fitted()
        X = self._validate_input(X)
        return self._compute_membership(X, self.result_.cluster_centers)

    def get_membership_matrix(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return membership matrix — symmetric API with KMeansClusterer.

        Parameters
        ----------
        X : optional array of shape (n_samples, n_features)
            If None, returns training membership matrix.
            If provided, computes memberships for new samples.

        Returns
        -------
        u : np.ndarray of shape (n_samples, n_clusters)
            Row sums ≈ 1.0 (fuzzy partition unity).
        """
        if X is None:
            self._check_fitted()
            return self.result_.membership_matrix
        return self.predict_proba(X)

    def get_uncertainty_mask(self, delta: float = 0.1) -> np.ndarray:
        """
        Return boolean mask of boundary / uncertain samples.

        A sample is 'uncertain' if the difference between its two highest
        membership values is less than delta. These are points that FCM
        assigns to the boundary of two clusters.

        Parameters
        ----------
        delta : float
            Threshold on membership gap. Default 0.1.

        Returns
        -------
        mask : np.ndarray of bool, shape (n_samples,)
        """
        self._check_fitted()
        u = self.result_.membership_matrix
        sorted_u = np.sort(u, axis=1)[:, ::-1]  # descending
        gap = sorted_u[:, 0] - sorted_u[:, 1]
        return gap < delta

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Return a flat diagnostics dictionary for dashboards and reports.

        Returns
        -------
        dict with keys:
            algorithm, n_clusters, m, max_iter, tol, random_state,
            n_iter, converged, fpc, fpe, fuzzy_sse, crisp_sse,
            execution_time_ms, cluster_sizes, cluster_proportions
        """
        self._check_fitted()
        r = self.result_
        return {
            "algorithm": "Fuzzy C-Means",
            "n_clusters": r.n_clusters,
            "m": r.m,
            "max_iter": r.max_iter,
            "tol": r.tol,
            "random_state": r.random_state,
            "n_iter": r.n_iter,
            "converged": r.converged,
            "fpc": r.fpc,
            "fpe": r.fpe,
            "fuzzy_sse": r.fuzzy_sse,
            "crisp_sse": r.crisp_sse,
            "execution_time_ms": r.execution_time_ms,
            "cluster_sizes": r.cluster_sizes,
            "cluster_proportions": r.cluster_proportions,
        }

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return crisp (defuzzified) labels."""
        self.fit(X)
        return self.result_.labels

    # ------------------------------------------------------------------
    # Core FCM math (private)
    # ------------------------------------------------------------------

    def _init_membership(self, n_samples: int) -> np.ndarray:
        """
        Initialize membership matrix U ∈ R^{n_samples × n_clusters}.

        Each row is a Dirichlet-like random draw normalized to sum to 1.
        This satisfies the fuzzy partition constraint at initialization.
        """
        rng = np.random.RandomState(self.random_state)
        # Uniform random, then row-normalize
        u = rng.uniform(low=0.0, high=1.0, size=(n_samples, self.n_clusters))
        row_sums = u.sum(axis=1, keepdims=True)
        return u / np.maximum(row_sums, 1e-12)

    def _compute_centers(self, X: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute fuzzy cluster centroids.

        v_j = sum_i( u_ij^m * x_i ) / sum_i( u_ij^m )

        Shape: (n_clusters, n_features)
        """
        um = u ** self.m                              # (n_samples, n_clusters)
        numerator = um.T @ X                          # (n_clusters, n_features)
        denominator = um.sum(axis=0).reshape(-1, 1)   # (n_clusters, 1)
        denominator = np.maximum(denominator, 1e-12)
        return numerator / denominator

    def _compute_membership(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Update membership matrix using the FCM update rule.

        u_ij = 1 / sum_l [ (d_ij / d_il)^{2/(m-1)} ]

        where d_ij = ||x_i - v_j||_2

        Special case: if x_i coincides exactly with center v_j, that
        sample gets u_ij = 1 and all other memberships = 0.

        Returns
        -------
        u_new : np.ndarray of shape (n_samples, n_clusters)
            Rows sum to 1.0 within floating-point precision.
        """
        # dists[i, j] = ||x_i - v_j||_2
        dists = cdist(X, centers, metric="euclidean")   # (n_samples, n_clusters)

        n_samples = X.shape[0]
        n_clusters = centers.shape[0]
        exponent = 2.0 / (self.m - 1.0)

        # Handle exact-coincidence rows (d_ij = 0 for some j)
        zero_mask = dists < 1e-12  # (n_samples, n_clusters) bool
        any_zero = zero_mask.any(axis=1)   # (n_samples,) bool

        # For normal rows: inv_power[i,j] = (1/d_ij)^exponent
        dists_safe = np.where(zero_mask, 1.0, dists)   # avoid division by zero
        inv_power = (1.0 / dists_safe) ** exponent      # (n_samples, n_clusters)

        row_sums = inv_power.sum(axis=1, keepdims=True)
        u_normal = inv_power / np.maximum(row_sums, 1e-12)

        # For coincident rows: 1 for the coincident cluster, 0 elsewhere
        u_coincident = zero_mask.astype(np.float64)

        # Combine: use coincident rule where any zero distance exists
        u_new = np.where(any_zero[:, np.newaxis], u_coincident, u_normal)

        # Numerical guard: ensure rows sum to 1 (handle fp drift)
        row_sums2 = u_new.sum(axis=1, keepdims=True)
        u_new = u_new / np.maximum(row_sums2, 1e-12)

        return u_new

    def _compute_objective(
        self, X: np.ndarray, centers: np.ndarray, u: np.ndarray
    ) -> float:
        """
        Compute fuzzy objective function J_m.

        J_m = sum_i sum_j ( u_ij^m * ||x_i - v_j||^2 )

        Returns
        -------
        float : J_m value.
        """
        dists_sq = cdist(X, centers, metric="sqeuclidean")   # (n_samples, n_clusters)
        um = u ** self.m
        return float(np.sum(um * dists_sq))

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
                "FuzzyCMeansClusterer is not fitted. Call fit(X) first."
            )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FuzzyCMeansClusterer("
            f"n_clusters={self.n_clusters}, m={self.m}, "
            f"max_iter={self.max_iter}, tol={self.tol}, "
            f"random_state={self.random_state})"
        )

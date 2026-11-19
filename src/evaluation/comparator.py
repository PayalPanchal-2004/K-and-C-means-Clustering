"""
Unified Comparison Engine — runs K-Means and Fuzzy C-Means on the same X
and returns a fully populated side-by-side metrics dictionary.

Usage
-----
from algorithms.kmeans import KMeansClusterer
from algorithms.fuzzy_cmeans import FuzzyCMeansClusterer
from src.evaluation.comparator import compare_algorithms

result = compare_algorithms(X, n_clusters=3)
print(result["summary_table"])   # list of dicts, one per algorithm
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from algorithms.kmeans import KMeansClusterer
from algorithms.fuzzy_cmeans import FuzzyCMeansClusterer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _internal_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    execution_time_ms: float,
    n_iter: int,
    converged: bool,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute internal validity metrics for a clustering result."""
    n_samples = len(X)
    n_unique = len(np.unique(labels))

    # Guard against degenerate single-cluster result
    if n_unique < 2:
        sil, dbi, chi = None, None, None
    else:
        sample_size = 5000 if n_samples > 10_000 else None
        sil = round(float(silhouette_score(X, labels, sample_size=sample_size, random_state=42)), 4)
        dbi = round(float(davies_bouldin_score(X, labels)), 4)
        chi = round(float(calinski_harabasz_score(X, labels)), 2)

    # SSE
    assigned = centers[labels]
    sse = round(float(np.sum((X - assigned) ** 2)), 4)
    rmse = round(float(np.sqrt(sse / n_samples)), 4)

    sizes = np.bincount(labels, minlength=centers.shape[0]).tolist()
    proportions = [round(s / n_samples, 4) for s in sizes]

    result = {
        "silhouette": sil,
        "davies_bouldin": dbi,
        "calinski_harabasz": chi,
        "sse": sse,
        "rmse": rmse,
        "execution_time_ms": round(execution_time_ms, 2),
        "n_iter": n_iter,
        "converged": converged,
        "cluster_sizes": sizes,
        "cluster_proportions": proportions,
        "n_samples": n_samples,
        "n_features": X.shape[1],
    }
    if extra:
        result.update(extra)
    return result


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------

def compare_algorithms(
    X: np.ndarray,
    n_clusters: int = 3,
    # K-Means params
    km_init: str = "k-means++",
    km_n_init: int = 10,
    km_max_iter: int = 300,
    km_tol: float = 1e-4,
    # FCM params
    fcm_m: float = 2.0,
    fcm_max_iter: int = 300,
    fcm_tol: float = 1e-4,
    # Common
    random_state: int = 42,
    dataset_name: str = "dataset",
) -> Dict[str, Any]:
    """
    Run both algorithms on identical X and collect side-by-side metrics.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Pre-scaled feature matrix. Must be identical for both algorithms.
    n_clusters : int
        Number of clusters used for both algorithms.
    km_init : str
        K-Means initialization strategy ('k-means++' or 'random').
    km_n_init : int
        K-Means number of restarts.
    km_max_iter : int
        K-Means maximum iterations.
    km_tol : float
        K-Means convergence tolerance.
    fcm_m : float
        FCM fuzzifier exponent (> 1.0).
    fcm_max_iter : int
        FCM maximum iterations.
    fcm_tol : float
        FCM convergence tolerance.
    random_state : int
        Seed for reproducibility (applied to both).
    dataset_name : str
        Human-readable label for reporting.

    Returns
    -------
    dict with keys:
        "dataset_name"   : str
        "n_clusters"     : int
        "X_shape"        : tuple
        "kmeans"         : dict of K-Means results + metrics
        "fcm"            : dict of FCM results + metrics
        "summary_table"  : list of two dicts (one per algorithm) for tabular display
        "centroid_shift" : dict with avg and max centroid shift between algorithms
        "winner"         : dict indicating which algorithm scored better per metric
    """
    X = np.asarray(X, dtype=np.float64)

    # ---- K-Means ----
    km = KMeansClusterer(
        n_clusters=n_clusters,
        init=km_init,
        n_init=km_n_init,
        max_iter=km_max_iter,
        tol=km_tol,
        random_state=random_state,
    )
    km.fit(X)
    km_r = km.result_

    km_metrics = _internal_metrics(
        X=X,
        labels=km_r.labels,
        centers=km_r.cluster_centers,
        execution_time_ms=km_r.execution_time_ms,
        n_iter=km_r.n_iter,
        converged=km_r.converged,
        extra={"inertia": km_r.inertia},
    )
    km_metrics["algorithm"] = "K-Means"
    km_metrics["init"] = km_init
    km_metrics["n_init"] = km_n_init
    km_metrics["fpc"] = None  # not applicable for K-Means
    km_metrics["fpe"] = None

    # ---- FCM ----
    fcm = FuzzyCMeansClusterer(
        n_clusters=n_clusters,
        m=fcm_m,
        max_iter=fcm_max_iter,
        tol=fcm_tol,
        random_state=random_state,
    )
    fcm.fit(X)
    fcm_r = fcm.result_

    fcm_metrics = _internal_metrics(
        X=X,
        labels=fcm_r.labels,
        centers=fcm_r.cluster_centers,
        execution_time_ms=fcm_r.execution_time_ms,
        n_iter=fcm_r.n_iter,
        converged=fcm_r.converged,
        extra={"fuzzy_sse": fcm_r.fuzzy_sse},
    )
    fcm_metrics["algorithm"] = "Fuzzy C-Means"
    fcm_metrics["m"] = fcm_m
    fcm_metrics["fpc"] = fcm_r.fpc
    fcm_metrics["fpe"] = fcm_r.fpe

    # ---- Centroid distance (after matching clusters by nearest-centre) ----
    centroid_shift = _centroid_shift(km_r.cluster_centers, fcm_r.cluster_centers)

    # ---- Winner determination (better = higher sil / chi, lower dbi / sse) ----
    winner = _determine_winner(km_metrics, fcm_metrics)

    # ---- Summary table ----
    summary_table = [
        {
            "Algorithm": "K-Means",
            "Silhouette ↑": km_metrics["silhouette"],
            "Davies-Bouldin ↓": km_metrics["davies_bouldin"],
            "Calinski-Harabasz ↑": km_metrics["calinski_harabasz"],
            "SSE ↓": km_metrics["sse"],
            "RMSE ↓": km_metrics["rmse"],
            "Iterations": km_metrics["n_iter"],
            "Converged": km_metrics["converged"],
            "Time (ms)": km_metrics["execution_time_ms"],
            "FPC ↑": km_metrics["fpc"],
            "Cluster Sizes": km_metrics["cluster_sizes"],
        },
        {
            "Algorithm": "Fuzzy C-Means",
            "Silhouette ↑": fcm_metrics["silhouette"],
            "Davies-Bouldin ↓": fcm_metrics["davies_bouldin"],
            "Calinski-Harabasz ↑": fcm_metrics["calinski_harabasz"],
            "SSE ↓": fcm_metrics["sse"],
            "RMSE ↓": fcm_metrics["rmse"],
            "Iterations": fcm_metrics["n_iter"],
            "Converged": fcm_metrics["converged"],
            "Time (ms)": fcm_metrics["execution_time_ms"],
            "FPC ↑": fcm_metrics["fpc"],
            "Cluster Sizes": fcm_metrics["cluster_sizes"],
        },
    ]

    return {
        "dataset_name": dataset_name,
        "n_clusters": n_clusters,
        "X_shape": X.shape,
        "kmeans": {**km_metrics, "model": km, "result": km_r},
        "fcm": {**fcm_metrics, "model": fcm, "result": fcm_r},
        "summary_table": summary_table,
        "centroid_shift": centroid_shift,
        "winner": winner,
    }


def _centroid_shift(
    km_centers: np.ndarray, fcm_centers: np.ndarray
) -> Dict[str, float]:
    """
    Compute pairwise centroid distances between K-Means and FCM centroids.

    Uses greedy nearest-centre matching (not full Hungarian matching).
    Returns avg and max shift.
    """
    from scipy.spatial.distance import cdist as _cdist
    D = _cdist(km_centers, fcm_centers, metric="euclidean")
    # Greedy match
    shifts = []
    used = set()
    for i in range(len(km_centers)):
        sorted_j = np.argsort(D[i])
        for j in sorted_j:
            if j not in used:
                shifts.append(D[i, j])
                used.add(j)
                break

    return {
        "avg_shift": round(float(np.mean(shifts)), 4),
        "max_shift": round(float(np.max(shifts)), 4),
        "per_cluster_shifts": [round(float(s), 4) for s in shifts],
    }


def _determine_winner(km: Dict, fcm: Dict) -> Dict[str, str]:
    """Return which algorithm wins each metric (or 'tie')."""
    winners: Dict[str, str] = {}

    def cmp_higher(key: str) -> str:
        a, b = km.get(key), fcm.get(key)
        if a is None or b is None:
            return "N/A"
        if a > b:
            return "K-Means"
        elif b > a:
            return "Fuzzy C-Means"
        return "Tie"

    def cmp_lower(key: str) -> str:
        a, b = km.get(key), fcm.get(key)
        if a is None or b is None:
            return "N/A"
        if a < b:
            return "K-Means"
        elif b < a:
            return "Fuzzy C-Means"
        return "Tie"

    winners["silhouette"] = cmp_higher("silhouette")
    winners["calinski_harabasz"] = cmp_higher("calinski_harabasz")
    winners["fpc"] = cmp_higher("fpc")          # FCM-specific, KM always None
    winners["davies_bouldin"] = cmp_lower("davies_bouldin")
    winners["sse"] = cmp_lower("sse")
    winners["rmse"] = cmp_lower("rmse")
    winners["execution_time_ms"] = cmp_lower("execution_time_ms")

    return winners

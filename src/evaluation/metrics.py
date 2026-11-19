"""Clustering Evaluation Metrics (Unsupervised Validity Measures)."""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def compute_sse(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """Compute within-cluster Sum of Squared Errors (SSE / Inertia)."""
    assigned_centers = centers[labels]
    return float(np.sum((X - assigned_centers) ** 2))


def compute_rmse(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """Compute Root Mean Square Error (RMSE) to cluster centroids as formulated in synopsis Chapter 5."""
    n_samples = len(X)
    sse = compute_sse(X, labels, centers)
    return float(np.sqrt(sse / n_samples))


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    execution_time_ms: float = 0.0,
    n_iter: int = 0,
    membership_matrix: Optional[np.ndarray] = None,
    fpc: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculate internal clustering validity metrics."""
    n_samples = len(X)
    n_unique_labels = len(np.unique(labels))

    # Guard against single-cluster edge cases
    if n_unique_labels < 2:
        return {
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
            "sse": compute_sse(X, labels, centers),
            "rmse": compute_rmse(X, labels, centers),
            "execution_time_ms": execution_time_ms,
            "n_iter": n_iter,
            "fpc": fpc,
            "cluster_sizes": [n_samples],
        }

    # Silhouette sampling if n_samples > 10,000 for efficiency
    sample_size = 5000 if n_samples > 10000 else None
    sil = float(silhouette_score(X, labels, sample_size=sample_size, random_state=42))
    dbi = float(davies_bouldin_score(X, labels))
    chi = float(calinski_harabasz_score(X, labels))

    sse = compute_sse(X, labels, centers)
    rmse = compute_rmse(X, labels, centers)
    cluster_counts = np.bincount(labels).tolist()

    if fpc is None and membership_matrix is not None:
        fpc = float(np.mean(np.sum(membership_matrix ** 2, axis=1)))

    return {
        "silhouette": round(sil, 4),
        "davies_bouldin": round(dbi, 4),
        "calinski_harabasz": round(chi, 2),
        "sse": round(sse, 2),
        "rmse": round(rmse, 4),
        "execution_time_ms": round(execution_time_ms, 2),
        "n_iter": n_iter,
        "fpc": round(fpc, 4) if fpc is not None else None,
        "cluster_sizes": cluster_counts,
    }

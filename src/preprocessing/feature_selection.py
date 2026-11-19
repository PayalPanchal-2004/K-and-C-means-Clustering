"""Re-export of feature selection utilities."""

from preprocessing.feature_selection import (
    remove_low_variance_features,
    detect_collinear_pairs,
    compute_pca_projection,
)

__all__ = [
    "remove_low_variance_features",
    "detect_collinear_pairs",
    "compute_pca_projection",
]

"""Feature selection and dimensionality analysis utilities for clustering."""

from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA


def remove_low_variance_features(
    df: pd.DataFrame,
    threshold: float = 0.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """Identify and remove features with variance below threshold."""
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df)
    kept_features = df.columns[selector.get_support()].tolist()
    dropped_features = [c for c in df.columns if c not in kept_features]
    return df[kept_features].copy(), dropped_features


def detect_collinear_pairs(
    df: pd.DataFrame,
    threshold: float = 0.85,
) -> List[Dict[str, Any]]:
    """Detect pairs of features with absolute Pearson correlation above threshold."""
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    collinear_pairs = []
    for col in upper.columns:
        high_corr = upper[col][upper[col] > threshold]
        for row_col, val in high_corr.items():
            collinear_pairs.append({
                "feature_1": row_col,
                "feature_2": col,
                "correlation": float(val),
            })
    return collinear_pairs


def compute_pca_projection(
    X: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, List[float], PCA]:
    """Project high-dimensional standardized array into 2D or 3D for visualization."""
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_.tolist()
    return coords, explained_var, pca

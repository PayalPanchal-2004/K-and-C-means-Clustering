"""
Phase 3 Algorithm Implementations.

Canonical implementations for fair algorithm comparison:
  - KMeansClusterer  : scikit-learn KMeans wrapper with full diagnostics
  - FuzzyCMeansClusterer : vectorized NumPy FCM (Bezdek, 1984)
"""

from .kmeans import KMeansClusterer
from .fuzzy_cmeans import FuzzyCMeansClusterer

__all__ = ["KMeansClusterer", "FuzzyCMeansClusterer"]

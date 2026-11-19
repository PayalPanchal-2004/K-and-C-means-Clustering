"""
Phase 3 Unit Tests — K-Means and Fuzzy C-Means Implementations.

Tests:
  1.  KMeans: basic fit on small synthetic data
  2.  KMeans: k-means++ initialization
  3.  KMeans: random initialization
  4.  KMeans: cluster labels range, centers shape, inertia positive
  5.  KMeans: membership matrix is binary one-hot and rows sum to 1
  6.  KMeans: reproducible results with same random_state
  7.  KMeans: predict on unseen data returns valid labels
  8.  KMeans: raises ValueError on bad inputs
  9.  FCM: basic fit on small synthetic data
  10. FCM: membership matrix rows sum to 1 (partition unity)
  11. FCM: FPC in valid range (1/n_clusters, 1]
  12. FCM: FPE non-negative
  13. FCM: objective decreases over iterations (monotone property)
  14. FCM: predict_proba rows sum to 1 on new data
  15. FCM: get_uncertainty_mask returns bool array
  16. FCM: raises ValueError for m <= 1.0
  17. FCM: reproduces with same random_state
  18. Cross: both algorithms fit all 5 real preprocessed datasets
  19. Cross: identical X inputs to both algorithms
  20. Cross: comparator produces expected keys and winner dict
  21. Cross: n_samples verified for all datasets
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algorithms.kmeans import KMeansClusterer
from algorithms.fuzzy_cmeans import FuzzyCMeansClusterer
from src.evaluation.comparator import compare_algorithms

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def blobs_small():
    """
    Three well-separated Gaussian blobs (60 samples × 4 features).
    Used for deterministic unit tests.
    """
    rng = np.random.RandomState(0)
    centers = np.array([[5, 5, 5, 5], [-5, -5, -5, -5], [5, -5, 5, -5]], dtype=float)
    X = np.vstack([
        rng.randn(20, 4) + centers[0],
        rng.randn(20, 4) + centers[1],
        rng.randn(20, 4) + centers[2],
    ])
    return X


@pytest.fixture(scope="session")
def preprocessed_datasets():
    """
    Load all 5 scaled datasets from data/processed/.
    Returns dict: name -> np.ndarray.
    """
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    files = {
        "credit_card_customers":          "credit_card_customers_scaled.csv",
        "customer_personality":           "customer_personality_scaled.csv",
        "wine_quality":                   "wine_quality_scaled.csv",
        "online_retail":                  "online_retail_scaled.csv",
        "synthetic_classification_1000":  "synthetic_classification_1000_scaled.csv",
    }
    datasets = {}
    for name, fname in files.items():
        path = os.path.join(processed_dir, fname)
        assert os.path.exists(path), f"Processed file missing: {path}"
        df = pd.read_csv(path)
        datasets[name] = df.to_numpy(dtype=np.float64)
    return datasets


# ===========================================================================
# K-Means Tests
# ===========================================================================

class TestKMeansClusterer:

    def test_basic_fit(self, blobs_small):
        """K-Means fits without error on well-separated blobs."""
        km = KMeansClusterer(n_clusters=3, random_state=42)
        km.fit(blobs_small)
        assert km.is_fitted_
        assert km.result_ is not None

    def test_labels_shape_and_range(self, blobs_small):
        """Labels array has correct shape and all values in [0, n_clusters)."""
        km = KMeansClusterer(n_clusters=3, random_state=42)
        km.fit(blobs_small)
        labels = km.result_.labels
        assert labels.shape == (blobs_small.shape[0],), "Labels shape mismatch"
        assert np.all(labels >= 0) and np.all(labels < 3), "Label values out of range"

    def test_cluster_centers_shape(self, blobs_small):
        """Cluster centers array has shape (n_clusters, n_features)."""
        km = KMeansClusterer(n_clusters=3, random_state=42)
        km.fit(blobs_small)
        centers = km.result_.cluster_centers
        assert centers.shape == (3, blobs_small.shape[1]), "Centers shape mismatch"

    def test_inertia_positive(self, blobs_small):
        """Inertia (SSE) must be a positive float."""
        km = KMeansClusterer(n_clusters=3, random_state=42)
        km.fit(blobs_small)
        assert km.result_.inertia > 0, "Inertia should be positive"

    def test_membership_matrix_binary_one_hot(self, blobs_small):
        """Hard membership matrix is binary one-hot and rows sum to 1."""
        km = KMeansClusterer(n_clusters=3, random_state=42)
        km.fit(blobs_small)
        u = km.get_membership_matrix()
        assert u.shape == (blobs_small.shape[0], 3)
        # All values must be 0 or 1
        assert np.all(np.isin(u, [0.0, 1.0])), "K-Means membership must be binary"
        # Each row sums to 1
        row_sums = u.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-9, err_msg="Row sums != 1")

    def test_kmeanspp_initialization(self, blobs_small):
        """K-Means++ runs without error and converges."""
        km = KMeansClusterer(n_clusters=3, init="k-means++", random_state=42)
        km.fit(blobs_small)
        assert km.result_.converged, "Should converge on well-separated blobs with k-means++"

    def test_random_initialization(self, blobs_small):
        """Random initialization runs without error."""
        km = KMeansClusterer(n_clusters=3, init="random", n_init=5, random_state=0)
        km.fit(blobs_small)
        assert km.is_fitted_

    def test_reproducibility(self, blobs_small):
        """Same random_state produces identical labels."""
        km1 = KMeansClusterer(n_clusters=3, random_state=99)
        km2 = KMeansClusterer(n_clusters=3, random_state=99)
        km1.fit(blobs_small)
        km2.fit(blobs_small)
        np.testing.assert_array_equal(
            km1.result_.labels, km2.result_.labels,
            err_msg="Same random_state must yield identical labels"
        )

    def test_predict_new_data(self, blobs_small):
        """predict() returns valid labels for held-out samples."""
        km = KMeansClusterer(n_clusters=3, random_state=42)
        km.fit(blobs_small[:40])  # train on 40
        preds = km.predict(blobs_small[40:])  # predict on 20
        assert preds.shape == (20,)
        assert np.all(preds >= 0) and np.all(preds < 3)

    def test_invalid_n_clusters(self, blobs_small):
        """Raises ValueError when n_clusters < 2."""
        with pytest.raises(ValueError, match="n_clusters"):
            KMeansClusterer(n_clusters=1)

    def test_invalid_init(self, blobs_small):
        """Raises ValueError for unknown init strategy."""
        with pytest.raises(ValueError, match="init"):
            KMeansClusterer(n_clusters=3, init="invalid_init")

    def test_nan_input_raises(self, blobs_small):
        """Raises ValueError if X contains NaN."""
        X_bad = blobs_small.copy()
        X_bad[0, 0] = np.nan
        km = KMeansClusterer(n_clusters=3)
        with pytest.raises(ValueError, match="NaN"):
            km.fit(X_bad)

    def test_diagnostics_keys(self, blobs_small):
        """get_diagnostics() returns all required keys."""
        km = KMeansClusterer(n_clusters=3, random_state=42)
        km.fit(blobs_small)
        d = km.get_diagnostics()
        required = {
            "algorithm", "n_clusters", "init", "n_init", "max_iter", "tol",
            "random_state", "n_iter", "converged", "inertia_sse",
            "execution_time_ms", "cluster_sizes", "cluster_proportions",
        }
        assert required.issubset(set(d.keys())), f"Missing keys: {required - set(d.keys())}"

    def test_cluster_sizes_sum_to_n(self, blobs_small):
        """Cluster sizes must sum to total sample count."""
        km = KMeansClusterer(n_clusters=3, random_state=42)
        km.fit(blobs_small)
        assert sum(km.result_.cluster_sizes) == blobs_small.shape[0]

    def test_not_fitted_raises(self):
        """Raises RuntimeError when predict is called before fit."""
        km = KMeansClusterer(n_clusters=3)
        with pytest.raises(RuntimeError, match="not fitted"):
            km.predict(np.zeros((5, 4)))


# ===========================================================================
# Fuzzy C-Means Tests
# ===========================================================================

class TestFuzzyCMeansClusterer:

    def test_basic_fit(self, blobs_small):
        """FCM fits without error on well-separated blobs."""
        fcm = FuzzyCMeansClusterer(n_clusters=3, m=2.0, random_state=42)
        fcm.fit(blobs_small)
        assert fcm.is_fitted_
        assert fcm.result_ is not None

    def test_labels_shape_and_range(self, blobs_small):
        """Defuzzified labels have correct shape and values in [0, n_clusters)."""
        fcm = FuzzyCMeansClusterer(n_clusters=3, m=2.0, random_state=42)
        fcm.fit(blobs_small)
        labels = fcm.result_.labels
        assert labels.shape == (blobs_small.shape[0],)
        assert np.all(labels >= 0) and np.all(labels < 3)

    def test_cluster_centers_shape(self, blobs_small):
        """Fuzzy centroids have shape (n_clusters, n_features)."""
        fcm = FuzzyCMeansClusterer(n_clusters=3, m=2.0, random_state=42)
        fcm.fit(blobs_small)
        assert fcm.result_.cluster_centers.shape == (3, blobs_small.shape[1])

    def test_membership_partition_unity(self, blobs_small):
        """Membership matrix rows sum to 1.0 (fuzzy partition unity constraint)."""
        fcm = FuzzyCMeansClusterer(n_clusters=3, m=2.0, random_state=42)
        fcm.fit(blobs_small)
        u = fcm.get_membership_matrix()
        assert u.shape == (blobs_small.shape[0], 3)
        row_sums = u.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, 1.0, atol=1e-6,
            err_msg="FCM membership rows must sum to 1.0 (partition unity)"
        )

    def test_membership_values_in_01(self, blobs_small):
        """All membership values are in [0, 1]."""
        fcm = FuzzyCMeansClusterer(n_clusters=3, m=2.0, random_state=42)
        fcm.fit(blobs_small)
        u = fcm.result_.membership_matrix
        assert np.all(u >= 0.0) and np.all(u <= 1.0 + 1e-9), \
            "Membership values must be in [0, 1]"

    def test_fpc_valid_range(self, blobs_small):
        """FPC must be in (1/n_clusters, 1]."""
        n_clusters = 3
        fcm = FuzzyCMeansClusterer(n_clusters=n_clusters, m=2.0, random_state=42)
        fcm.fit(blobs_small)
        fpc = fcm.result_.fpc
        lower_bound = 1.0 / n_clusters
        assert fpc > lower_bound, f"FPC={fpc} should be > 1/k={lower_bound}"
        assert fpc <= 1.0 + 1e-9, f"FPC={fpc} should be <= 1"

    def test_fpe_non_negative(self, blobs_small):
        """Fuzzy Partition Entropy must be non-negative."""
        fcm = FuzzyCMeansClusterer(n_clusters=3, m=2.0, random_state=42)
        fcm.fit(blobs_small)
        assert fcm.result_.fpe >= 0.0, "FPE must be non-negative"

    def test_objective_decreases(self, blobs_small):
        """Fuzzy objective J_m must be monotonically non-increasing."""
        fcm = FuzzyCMeansClusterer(n_clusters=3, m=2.0, max_iter=200, random_state=42)
        fcm.fit(blobs_small)
        history = fcm.result_.objective_history
        assert len(history) >= 1, "objective_history must be populated"
        for i in range(1, len(history)):
            assert history[i] <= history[i - 1] + 1e-6, \
                f"Objective increased at iteration {i}: {history[i-1]} -> {history[i]}"

    def test_predict_proba_partition_unity(self, blobs_small):
        """predict_proba rows sum to 1 on unseen data."""
        fcm = FuzzyCMeansClusterer(n_clusters=3, m=2.0, random_state=42)
        fcm.fit(blobs_small[:40])
        proba = fcm.predict_proba(blobs_small[40:])
        assert proba.shape == (20, 3)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_uncertainty_mask_dtype(self, blobs_small):
        """get_uncertainty_mask returns a boolean array of correct shape."""
        fcm = FuzzyCMeansClusterer(n_clusters=3, m=2.0, random_state=42)
        fcm.fit(blobs_small)
        mask = fcm.get_uncertainty_mask(delta=0.1)
        assert mask.dtype == bool
        assert mask.shape == (blobs_small.shape[0],)

    def test_m_too_small_raises(self):
        """Raises ValueError for m <= 1.0."""
        with pytest.raises(ValueError, match="m.*1.0"):
            FuzzyCMeansClusterer(n_clusters=3, m=1.0)
        with pytest.raises(ValueError, match="m.*1.0"):
            FuzzyCMeansClusterer(n_clusters=3, m=0.5)

    def test_reproducibility(self, blobs_small):
        """Same random_state produces identical membership matrix."""
        fcm1 = FuzzyCMeansClusterer(n_clusters=3, m=2.0, random_state=7)
        fcm2 = FuzzyCMeansClusterer(n_clusters=3, m=2.0, random_state=7)
        fcm1.fit(blobs_small)
        fcm2.fit(blobs_small)
        np.testing.assert_allclose(
            fcm1.result_.membership_matrix,
            fcm2.result_.membership_matrix,
            atol=1e-10,
            err_msg="Same random_state must produce identical membership matrix"
        )

    def test_different_m_produces_different_results(self, blobs_small):
        """Different fuzzifier values produce different FPC values."""
        fcm_soft = FuzzyCMeansClusterer(n_clusters=3, m=3.0, random_state=42)
        fcm_hard = FuzzyCMeansClusterer(n_clusters=3, m=1.5, random_state=42)
        fcm_soft.fit(blobs_small)
        fcm_hard.fit(blobs_small)
        # Higher m → softer memberships → lower FPC
        assert fcm_soft.result_.fpc < fcm_hard.result_.fpc, \
            "Higher m should yield lower FPC (softer partitions)"

    def test_nan_input_raises(self, blobs_small):
        """Raises ValueError if X contains NaN."""
        X_bad = blobs_small.copy()
        X_bad[5, 2] = np.nan
        fcm = FuzzyCMeansClusterer(n_clusters=3, m=2.0)
        with pytest.raises(ValueError, match="NaN"):
            fcm.fit(X_bad)

    def test_diagnostics_keys(self, blobs_small):
        """get_diagnostics() returns all required keys."""
        fcm = FuzzyCMeansClusterer(n_clusters=3, m=2.0, random_state=42)
        fcm.fit(blobs_small)
        d = fcm.get_diagnostics()
        required = {
            "algorithm", "n_clusters", "m", "max_iter", "tol", "random_state",
            "n_iter", "converged", "fpc", "fpe", "fuzzy_sse", "crisp_sse",
            "execution_time_ms", "cluster_sizes", "cluster_proportions",
        }
        assert required.issubset(set(d.keys())), f"Missing keys: {required - set(d.keys())}"

    def test_not_fitted_raises(self):
        """Raises RuntimeError when predict called before fit."""
        fcm = FuzzyCMeansClusterer(n_clusters=3, m=2.0)
        with pytest.raises(RuntimeError, match="not fitted"):
            fcm.predict(np.zeros((5, 4)))

    def test_cluster_sizes_sum_to_n(self, blobs_small):
        """Cluster sizes must sum to total sample count."""
        fcm = FuzzyCMeansClusterer(n_clusters=3, m=2.0, random_state=42)
        fcm.fit(blobs_small)
        assert sum(fcm.result_.cluster_sizes) == blobs_small.shape[0]


# ===========================================================================
# Cross-Algorithm & Real Dataset Tests
# ===========================================================================

class TestCrossAlgorithmAndRealDatasets:

    # Expected minimum rows per dataset
    EXPECTED_SHAPES = {
        "credit_card_customers":          (10127, 1),
        "customer_personality":           (2240,  1),
        "wine_quality":                   (6497,  1),
        "online_retail":                  (4000,  1),  # min 4000 after RFM aggregation
        "synthetic_classification_1000":  (1000, 230),
    }

    def test_preprocessed_files_exist(self, preprocessed_datasets):
        """All 5 processed dataset files exist and load correctly."""
        assert len(preprocessed_datasets) == 5

    @pytest.mark.parametrize("name", [
        "credit_card_customers",
        "customer_personality",
        "wine_quality",
        "online_retail",
        "synthetic_classification_1000",
    ])
    def test_dataset_minimum_rows(self, preprocessed_datasets, name):
        """Each dataset meets minimum row count requirement."""
        X = preprocessed_datasets[name]
        min_rows, min_cols = self.EXPECTED_SHAPES[name]
        assert X.shape[0] >= min_rows, \
            f"{name}: expected >= {min_rows} rows, got {X.shape[0]}"
        assert X.shape[1] >= min_cols, \
            f"{name}: expected >= {min_cols} cols, got {X.shape[1]}"

    @pytest.mark.parametrize("name", [
        "credit_card_customers",
        "customer_personality",
        "wine_quality",
        "online_retail",
        "synthetic_classification_1000",
    ])
    def test_kmeans_fits_real_dataset(self, preprocessed_datasets, name):
        """K-Means fits and returns valid results on each real dataset."""
        X = preprocessed_datasets[name]
        km = KMeansClusterer(n_clusters=3, random_state=42)
        km.fit(X)
        assert km.is_fitted_
        assert len(np.unique(km.result_.labels)) >= 2, \
            "K-Means should produce at least 2 non-empty clusters"

    @pytest.mark.parametrize("name", [
        "credit_card_customers",
        "customer_personality",
        "wine_quality",
        "online_retail",
        "synthetic_classification_1000",
    ])
    def test_fcm_fits_real_dataset(self, preprocessed_datasets, name):
        """FCM fits and returns valid results on each real dataset."""
        X = preprocessed_datasets[name]
        fcm = FuzzyCMeansClusterer(n_clusters=3, m=2.0, random_state=42)
        fcm.fit(X)
        assert fcm.is_fitted_
        # Partition unity on real data
        row_sums = fcm.result_.membership_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5,
            err_msg=f"{name}: membership rows must sum to 1")

    def test_identical_input_guarantee(self, preprocessed_datasets):
        """
        K-Means and FCM receive exactly the same X array — verified by
        checking that fitting both on the same ndarray reference produces
        no data mutation.
        """
        X = preprocessed_datasets["wine_quality"].copy()
        X_snapshot = X.copy()

        km = KMeansClusterer(n_clusters=3, random_state=42)
        km.fit(X)
        np.testing.assert_array_equal(X, X_snapshot, err_msg="K-Means mutated X")

        fcm = FuzzyCMeansClusterer(n_clusters=3, m=2.0, random_state=42)
        fcm.fit(X)
        np.testing.assert_array_equal(X, X_snapshot, err_msg="FCM mutated X")

    def test_comparator_keys(self, preprocessed_datasets):
        """compare_algorithms() returns all required top-level keys."""
        X = preprocessed_datasets["customer_personality"]
        result = compare_algorithms(X, n_clusters=3, dataset_name="customer_personality")

        required_keys = {
            "dataset_name", "n_clusters", "X_shape",
            "kmeans", "fcm", "summary_table",
            "centroid_shift", "winner",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_comparator_winner_dict_keys(self, preprocessed_datasets):
        """Winner dict contains metric comparison keys."""
        X = preprocessed_datasets["wine_quality"]
        result = compare_algorithms(X, n_clusters=3)
        winner = result["winner"]
        expected_winner_keys = {
            "silhouette", "calinski_harabasz", "davies_bouldin",
            "sse", "rmse", "execution_time_ms",
        }
        assert expected_winner_keys.issubset(set(winner.keys()))

    def test_comparator_summary_table_has_two_rows(self, preprocessed_datasets):
        """Summary table has exactly two rows (one per algorithm)."""
        X = preprocessed_datasets["online_retail"]
        result = compare_algorithms(X, n_clusters=3)
        assert len(result["summary_table"]) == 2

    def test_synthetic_dataset_uses_230_features(self, preprocessed_datasets):
        """Synthetic dataset has 1000 rows; 230 feature cols F1-F230 (plus possible index col)."""
        X = preprocessed_datasets["synthetic_classification_1000"]
        assert X.shape[0] == 1000, \
            f"Synthetic dataset must have 1000 rows, got {X.shape[0]}"
        # Accepts 230 (pure features) or 231 (230 features + pandas index col)
        assert X.shape[1] in (230, 231), \
            f"Synthetic dataset must have 230 or 231 columns, got {X.shape[1]}"

    def test_fcm_membership_soft_on_wine(self, preprocessed_datasets):
        """FCM membership values on Wine Quality are genuinely soft (not all 0/1)."""
        X = preprocessed_datasets["wine_quality"]
        fcm = FuzzyCMeansClusterer(n_clusters=3, m=2.0, random_state=42)
        fcm.fit(X)
        u = fcm.result_.membership_matrix
        # At least some memberships are strictly between 0 and 1
        soft = np.sum((u > 0.01) & (u < 0.99))
        assert soft > 0, "FCM should produce soft (non-binary) membership values"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

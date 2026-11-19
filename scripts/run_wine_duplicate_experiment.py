"""Phase 2 Experiment: Wine Quality Duplicate Analysis.

Compares K-Means and Fuzzy C-Means clustering on:
- Experiment A: Original Dataset (6,497 records, retaining 1,177 duplicates)
- Experiment B: Deduplicated Dataset (5,320 unique records)

Evaluates:
- Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score, SSE, RMSE
- Execution time (ms), Convergence Iterations, Cluster Size Distributions
- Centroid gravitation and mathematical impact of duplicate observation density
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset_loader import RAW_DATA_DIR, PROCESSED_DATA_DIR, smart_read_table
from preprocessing.data_preprocessing import preprocess_wine_quality
from src.models.kmeans_model import KMeansModel
from src.models.fcm_model import FuzzyCMeansModel
from src.evaluation.metrics import evaluate_clustering


def run_experiment():
    print("=" * 90)
    print("PHASE 2 SPECIAL EXPERIMENT: WINE QUALITY DUPLICATE ANALYSIS")
    print("Comparison of Clustering Performance on Original vs. Deduplicated Manifolds")
    print("=" * 90)

    # 1. Load Raw Data
    raw_path = os.path.join(RAW_DATA_DIR, "wine_quality", "winequality.csv")
    raw_df = smart_read_table(raw_path)
    total_raw = len(raw_df)
    n_duplicates = int(raw_df.duplicated().sum())
    n_unique = total_raw - n_duplicates

    print(f"\n📂 Raw Dataset Loaded: {total_raw:,} records")
    print(f"   • Exact Duplicate Observations: {n_duplicates:,} ({n_duplicates/total_raw*100:.2f}%)")
    print(f"   • Unique Observations: {n_unique:,} ({n_unique/total_raw*100:.2f}%)")

    # 2. Preprocess Both Versions
    print("\n⚙️  Standardizing Experiment A (Original: 6,497 records)...")
    X_exp_a, df_exp_a, summary_a = preprocess_wine_quality(raw_df, deduplicate=False)
    exp_a_path = os.path.join(PROCESSED_DATA_DIR, "wine_quality_scaled.csv")
    df_exp_a.to_csv(exp_a_path, index=False)
    print(f"   💾 Saved Experiment A -> {exp_a_path}")

    print("\n⚙️  Standardizing Experiment B (Deduplicated: 5,320 records)...")
    X_exp_b, df_exp_b, summary_b = preprocess_wine_quality(raw_df, deduplicate=True)
    exp_b_path = os.path.join(PROCESSED_DATA_DIR, "wine_quality_dedup_scaled.csv")
    df_exp_b.to_csv(exp_b_path, index=False)
    print(f"   💾 Saved Experiment B -> {exp_b_path}")

    # 3. Clustering Execution Configuration
    K = 3
    M = 2.0
    RANDOM_SEED = 42
    MAX_ITER = 300
    TOL = 1e-4

    results = {}

    # -------------------------------------------------------------
    # Experiment A: K-Means & FCM
    # -------------------------------------------------------------
    print("\n🔬 Executing Experiment A (Original with Duplicates: N=6,497)...")
    km_a = KMeansModel(n_clusters=K, max_iter=MAX_ITER, tol=TOL, random_state=RANDOM_SEED)
    km_a.fit(X_exp_a)
    metrics_km_a = evaluate_clustering(
        X_exp_a, km_a.labels_, km_a.cluster_centers_, km_a.execution_time_ms_, km_a.n_iter_
    )
    results["KM_ExpA"] = (km_a, metrics_km_a)

    fcm_a = FuzzyCMeansModel(n_clusters=K, m=M, max_iter=MAX_ITER, tol=TOL, random_state=RANDOM_SEED)
    fcm_a.fit(X_exp_a)
    metrics_fcm_a = evaluate_clustering(
        X_exp_a, fcm_a.labels_, fcm_a.cluster_centers_, fcm_a.execution_time_ms_, fcm_a.n_iter_,
        membership_matrix=fcm_a.u_, fpc=fcm_a.fpc_
    )
    results["FCM_ExpA"] = (fcm_a, metrics_fcm_a)

    # -------------------------------------------------------------
    # Experiment B: K-Means & FCM
    # -------------------------------------------------------------
    print("🔬 Executing Experiment B (Deduplicated: N=5,320)...")
    km_b = KMeansModel(n_clusters=K, max_iter=MAX_ITER, tol=TOL, random_state=RANDOM_SEED)
    km_b.fit(X_exp_b)
    metrics_km_b = evaluate_clustering(
        X_exp_b, km_b.labels_, km_b.cluster_centers_, km_b.execution_time_ms_, km_b.n_iter_
    )
    results["KM_ExpB"] = (km_b, metrics_km_b)

    fcm_b = FuzzyCMeansModel(n_clusters=K, m=M, max_iter=MAX_ITER, tol=TOL, random_state=RANDOM_SEED)
    fcm_b.fit(X_exp_b)
    metrics_fcm_b = evaluate_clustering(
        X_exp_b, fcm_b.labels_, fcm_b.cluster_centers_, fcm_b.execution_time_ms_, fcm_b.n_iter_,
        membership_matrix=fcm_b.u_, fpc=fcm_b.fpc_
    )
    results["FCM_ExpB"] = (fcm_b, metrics_fcm_b)

    # 4. Comparative Metrics Table
    print("\n" + "=" * 90)
    print("EXPERIMENTAL RESULTS: WINE QUALITY (ORIGINAL VS. DEDUPLICATED)")
    print("=" * 90)

    header = f"{'Algorithm & Version':<26} | {'N Samples':<9} | {'Silhouette':<10} | {'DB Index':<8} | {'CH Index':<9} | {'SSE':<10} | {'RMSE':<8} | {'Time(ms)':<8} | {'Iter'}"
    print(header)
    print("-" * len(header))

    rows = [
        ("K-Means (Original)", 6497, metrics_km_a),
        ("K-Means (Deduplicated)", 5320, metrics_km_b),
        ("FCM (Original, m=2.0)", 6497, metrics_fcm_a),
        ("FCM (Deduplicated, m=2.0)", 5320, metrics_fcm_b),
    ]

    for name, n_s, m in rows:
        print(
            f"{name:<26} | {n_s:<9,} | {m['silhouette']:<10.4f} | {m['davies_bouldin']:<8.4f} | "
            f"{m['calinski_harabasz']:<9.1f} | {m['sse']:<10.1f} | {m['rmse']:<8.4f} | "
            f"{m['execution_time_ms']:<8.2f} | {m['n_iter']}"
        )

    # 5. Cluster Size & Assignment Shifts
    print("\n" + "=" * 90)
    print("CLUSTER SIZE DISTRIBUTIONS & PERCENTAGES")
    print("=" * 90)

    def print_sizes(name, sizes, total):
        pcts = [f"{s/total*100:.1f}%" for s in sizes]
        print(f"   • {name:<26}: {sizes} -> ({', '.join(pcts)})")

    print_sizes("K-Means (Original)", metrics_km_a["cluster_sizes"], 6497)
    print_sizes("K-Means (Deduplicated)", metrics_km_b["cluster_sizes"], 5320)
    print_sizes("FCM (Original)", metrics_fcm_a["cluster_sizes"], 6497)
    print_sizes("FCM (Deduplicated)", metrics_fcm_b["cluster_sizes"], 5320)

    if metrics_fcm_a["fpc"] and metrics_fcm_b["fpc"]:
        print(f"\n   FCM Partition Coefficient (FPC): Original = {metrics_fcm_a['fpc']:.4f} | Deduplicated = {metrics_fcm_b['fpc']:.4f}")

    # 6. Centroid Shift Analysis
    print("\n" + "=" * 90)
    print("CENTROID STABILITY & GEOMETRIC SHIFT ANALYSIS")
    print("=" * 90)
    # Match centroids between Exp A and Exp B by minimum Euclidean distance
    from scipy.spatial.distance import cdist
    dists_km = cdist(km_a.cluster_centers_, km_b.cluster_centers_)
    min_shifts_km = np.min(dists_km, axis=1)
    print(f"   • K-Means Average Centroid Shift (L2 distance): {np.mean(min_shifts_km):.4f} (Max shift: {np.max(min_shifts_km):.4f})")

    dists_fcm = cdist(fcm_a.cluster_centers_, fcm_b.cluster_centers_)
    min_shifts_fcm = np.min(dists_fcm, axis=1)
    print(f"   • FCM Average Centroid Shift (L2 distance):     {np.mean(min_shifts_fcm):.4f} (Max shift: {np.max(min_shifts_fcm):.4f})")

    return results


if __name__ == "__main__":
    run_experiment()

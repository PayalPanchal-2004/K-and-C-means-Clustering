"""
Phase 3 Integration Smoke Test.
Runs both algorithms on all 5 preprocessed datasets and prints a comparison table.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from algorithms.kmeans import KMeansClusterer
from algorithms.fuzzy_cmeans import FuzzyCMeansClusterer
from src.evaluation.comparator import compare_algorithms

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

DATASETS = {
    "Credit Card Customers":          "credit_card_customers_scaled.csv",
    "Customer Personality":           "customer_personality_scaled.csv",
    "Wine Quality":                   "wine_quality_scaled.csv",
    "Online Retail (RFM)":            "online_retail_scaled.csv",
    "Synthetic Classification 1000":  "synthetic_classification_1000_scaled.csv",
}

N_CLUSTERS = 3
RANDOM_STATE = 42

print("=" * 90)
print("PHASE 3 — Algorithm Comparison Smoke Test")
print(f"n_clusters = {N_CLUSTERS}  |  random_state = {RANDOM_STATE}  |  FCM m = 2.0")
print("=" * 90)

rows = []

for name, fname in DATASETS.items():
    path = os.path.join(PROCESSED_DIR, fname)
    if not os.path.exists(path):
        print(f"[SKIP] {name} — file not found: {path}")
        continue

    X = pd.read_csv(path).to_numpy(dtype=np.float64)
    print(f"\n▶  {name}  |  shape: {X.shape}")

    result = compare_algorithms(
        X,
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        dataset_name=name,
        fcm_m=2.0,
    )

    for row in result["summary_table"]:
        rows.append({
            "Dataset":            name,
            "Algorithm":          row["Algorithm"],
            "Silhouette":         row["Silhouette ↑"],
            "DB Index":           row["Davies-Bouldin ↓"],
            "CH Index":           row["Calinski-Harabasz ↑"],
            "SSE":                row["SSE ↓"],
            "RMSE":               row["RMSE ↓"],
            "Iterations":         row["Iterations"],
            "Converged":          row["Converged"],
            "Time (ms)":          row["Time (ms)"],
            "FPC":                row["FPC ↑"],
            "Cluster Sizes":      str(row["Cluster Sizes"]),
        })

    w = result["winner"]
    cs = result["centroid_shift"]
    print(f"   Winner — Silhouette: {w['silhouette']}  |  DB Index: {w['davies_bouldin']}  "
          f"|  CH: {w['calinski_harabasz']}  |  SSE: {w['sse']}")
    print(f"   Centroid Shift (avg / max): {cs['avg_shift']} / {cs['max_shift']}")

print("\n")
print("=" * 90)
print("FULL COMPARISON TABLE")
print("=" * 90)
df = pd.DataFrame(rows)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.4f}".format)
print(df.to_string(index=False))
print("\n✅  Phase 3 smoke test complete.")

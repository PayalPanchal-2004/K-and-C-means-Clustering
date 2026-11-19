"""Runner script to execute, benchmark, and print summaries for all 5 preprocessed datasets."""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset_loader import (
    DATASET_CATALOG,
    smart_read_table,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)
from preprocessing.data_preprocessing import (
    preprocess_dataset_by_id,
    PREPROCESSOR_MAPPING,
)


def run_pipeline():
    print("=" * 85)
    print("PHASE 2: DATA PREPROCESSING AND EXPLORATORY DATA ANALYSIS PIPELINE")
    print("=" * 85)

    dataset_raw_files = {
        "credit_card_customers": os.path.join(RAW_DATA_DIR, "credit_card_customers", "BankChurners.csv"),
        "customer_personality": os.path.join(RAW_DATA_DIR, "customer_personality", "marketing_campaign.csv"),
        "wine_quality": os.path.join(RAW_DATA_DIR, "wine_quality", "winequality.csv"),
        "online_retail": os.path.join(RAW_DATA_DIR, "online_retail", "Online Retail.xlsx"),
        "synthetic_classification_1000": os.path.join(RAW_DATA_DIR, "synthetic_classification_1000", "data_train_1000.csv"),
    }

    all_summaries = {}

    for d_id, raw_file in dataset_raw_files.items():
        spec = DATASET_CATALOG[d_id]
        print(f"\n⚙️  Processing Dataset: {spec.name}...")
        raw_df = smart_read_table(raw_file)
        
        X_scaled, scaled_df, summary = preprocess_dataset_by_id(d_id, raw_df)
        all_summaries[d_id] = summary

        # Persist scaled data
        scaled_csv_path = os.path.join(PROCESSED_DATA_DIR, f"{d_id}_scaled.csv")
        scaled_df.to_csv(scaled_csv_path, index=True)
        print(f"   💾 Saved standardized feature matrix -> {scaled_csv_path}")

    print("\n" + "=" * 85)
    print("DATASET PREPROCESSING BEFORE-AND-AFTER COMPARISON TABLE")
    print("=" * 85)

    header = f"{'Dataset Name':<32} | {'Before Shape':<14} | {'After Shape':<14} | {'Missing':<10} | {'Outliers(IQR)':<14} | {'Valid'}"
    print(header)
    print("-" * len(header))

    for d_id, summ in all_summaries.items():
        b_shape = f"{summ.original_shape[0]:,} x {summ.original_shape[1]}"
        a_shape = f"{summ.processed_shape[0]:,} x {summ.processed_shape[1]}"
        missing = f"{sum(summ.missing_values_before.values())} -> {summ.missing_values_after}"
        outliers = f"{sum(summ.outlier_counts_iqr.values()):,}"
        valid = "✅ PASS" if summ.is_valid else "❌ FAIL"
        print(f"{summ.dataset_name[:32]:<32} | {b_shape:<14} | {a_shape:<14} | {missing:<10} | {outliers:<14} | {valid}")

    print("\n" + "=" * 85)
    print("DETAILED DATASET-WISE PREPROCESSING SUMMARIES")
    print("=" * 85)

    for d_id, summ in all_summaries.items():
        print(f"\n📌 [{summ.dataset_name}]")
        print(f"   • Raw Dimensions: {summ.original_shape[0]:,} rows × {summ.original_shape[1]} columns")
        print(f"   • Preprocessed Clustering Dimensions: {summ.processed_shape[0]:,} rows × {summ.processed_shape[1]} features")
        print(f"   • Excluded Identifiers: {summ.excluded_identifiers or 'None'}")
        print(f"   • Excluded Targets: {summ.excluded_targets or 'None'}")
        print(f"   • Excluded Auxiliary / Redundant Columns: {summ.excluded_auxiliary or 'None'}")
        print(f"   • Duplicates in Raw Data: {summ.duplicates_detected:,}")
        print(f"   • Total Missing Values Handled: {sum(summ.missing_values_before.values()):,}")
        print(f"   • Outliers Detected via IQR: {sum(summ.outlier_counts_iqr.values()):,}")
        print(f"   • Outliers Detected via Z-Score (|z|>3): {sum(summ.outlier_counts_zscore.values()):,}")
        print(f"   • Features Scaled with StandardScaler: {len(summ.final_feature_names)} features (mean=0, std=1)")
        print(f"   • Feature Sample: {summ.final_feature_names[:6]} ... {summ.final_feature_names[-3:]}")
        print(f"   • Validation Status: {'VALIDATED' if summ.is_valid else 'FAILED'}")
        for note in summ.validation_notes:
            print(f"     - {note}")


if __name__ == "__main__":
    run_pipeline()

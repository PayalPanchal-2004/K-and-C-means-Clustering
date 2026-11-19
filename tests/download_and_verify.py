"""Script to download, load, and verify the five Kaggle datasets."""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset_loader import (
    DATASET_CATALOG,
    load_credit_card_customers,
    load_customer_personality,
    load_wine_quality,
    load_online_retail,
    load_synthetic_classification,
    validate_all_datasets,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)

def run():
    print("=" * 80)
    print("PHASE 1: DATASET DOWNLOADING, LOADING, AND VALIDATION")
    print("=" * 80)
    
    report = validate_all_datasets()
    
    print("\n" + "=" * 80)
    print("DATASET INTEGRATION SUMMARY REPORT")
    print("=" * 80)
    
    for d_id, res in report.items():
        spec = DATASET_CATALOG[d_id]
        print(f"\n📁 Dataset: {spec.name} ({d_id})")
        print(f"   URL: {spec.kaggle_url}")
        print(f"   Status: {res['status']}")
        
        if res["status"] == "LOADED_SUCCESSFULLY":
            summary = res["summary"]
            print(f"   ✅ Raw Record Count: {summary['raw_rows']:,}")
            print(f"   ✅ Raw Column Count: {summary['raw_cols']}")
            print(f"   ✅ Processed Clustering Record Count: {summary['processed_rows']:,}")
            print(f"   ✅ Clustering Feature Count: {summary['clustering_features']}")
            print(f"   ✅ Excluded Identifiers: {summary['excluded_identifiers']}")
            print(f"   ✅ Excluded Target: {summary['excluded_target']}")
            print(f"   ✅ Raw Missing Values: {summary['missing_values_raw']:,}")
            print(f"   ✅ Raw Duplicate Rows: {summary['duplicate_rows_raw']:,}")
            if "original_transaction_count" in summary:
                print(f"   📊 [Online Retail Transaction Count]: {summary['original_transaction_count']:,}")
                print(f"   📊 [Online Retail Unique Customers]: {summary['unique_customer_count_aggregated']:,}")
            if "combined_wine_count_verified" in summary:
                print(f"   🍷 [Combined Wine Count Verified]: {summary['combined_wine_count_verified']} ({summary['raw_rows']:,} rows)")
        else:
            print(f"   ❌ Error: {res['error']}")
            instructions = res["manual_instructions"]
            print(f"   📥 Manual Download Instructions:")
            print(f"      1. Download from: {instructions['kaggle_url']}")
            print(f"      2. Expected file(s): {instructions['expected_files']}")
            print(f"      3. Place inside: {instructions['destination_folder']}/")

if __name__ == "__main__":
    run()

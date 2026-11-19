"""Unit tests for Dataset Loader and Preprocessing."""

import os
import pytest
import pandas as pd
from src.data.dataset_loader import (
    DATASET_CATALOG,
    load_credit_card_customers,
    load_customer_personality,
    load_wine_quality,
    load_online_retail,
    load_synthetic_classification,
    smart_read_table,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)


def test_catalog_specifications():
    """Verify all five required datasets are registered."""
    expected_keys = [
        "credit_card_customers",
        "customer_personality",
        "wine_quality",
        "online_retail",
        "synthetic_classification_1000",
    ]
    for key in expected_keys:
        assert key in DATASET_CATALOG
        spec = DATASET_CATALOG[key]
        assert spec.min_required_records >= 1000
        assert spec.kaggle_url.startswith("https://www.kaggle.com/")


def test_credit_card_customers():
    """Verify Credit Card Customers dataset loading, shape, and exclusions."""
    raw_df, feature_df, summary = load_credit_card_customers()
    assert summary["raw_rows"] == 10127
    assert summary["raw_rows"] >= 1000
    assert summary["processed_rows"] == 10127
    assert "CLIENTNUM" not in feature_df.columns
    assert "Attrition_Flag" not in feature_df.columns
    assert feature_df.isnull().sum().sum() == 0


def test_customer_personality():
    """Verify Customer Personality dataset loading, shape, and exclusions."""
    raw_df, feature_df, summary = load_customer_personality()
    assert summary["raw_rows"] == 2240
    assert summary["raw_rows"] >= 1000
    assert summary["processed_rows"] == 2240
    assert "ID" not in feature_df.columns
    assert "Response" not in feature_df.columns
    assert summary["missing_values_raw"] == 24  # Known Income missing count
    assert feature_df.isnull().sum().sum() == 0


def test_wine_quality_combined():
    """Verify Wine Quality dataset combined count of 6,497 and target exclusions."""
    raw_df, feature_df, summary = load_wine_quality()
    assert summary["raw_rows"] == 6497
    assert summary["combined_wine_count_verified"] is True
    assert summary["processed_rows"] == 6497
    assert "quality" not in feature_df.columns
    assert "good" not in feature_df.columns
    assert "color" not in feature_df.columns
    assert feature_df.shape[1] == 11  # 11 physicochemical features
    assert feature_df.isnull().sum().sum() == 0


def test_online_retail_aggregation():
    """Verify Online Retail transaction count and customer RFM aggregation."""
    raw_df, feature_df, summary = load_online_retail()
    assert summary["original_transaction_count"] == 541909
    assert summary["unique_customer_count_aggregated"] >= 4000
    assert summary["processed_rows"] == 4338
    assert "CustomerID" not in feature_df.columns
    assert list(feature_df.columns) == ["Recency", "Frequency", "Monetary"]
    assert feature_df.isnull().sum().sum() == 0


def test_synthetic_classification():
    """Verify Synthetic Classification 1000 dataset loading or informative pending state."""
    synthetic_dir = os.path.join(RAW_DATA_DIR, "synthetic_classification_1000")
    file_candidates = ["data_train_1000.csv", "train.csv"]
    file_exists = any(os.path.exists(os.path.join(synthetic_dir, f)) for f in file_candidates)

    if file_exists:
        raw_df, feature_df, summary = load_synthetic_classification()
        assert summary["raw_rows"] == 1000
        assert summary["raw_cols"] == 231
        assert summary["processed_rows"] == 1000
        assert summary["clustering_features"] == 230
        assert "Label" not in feature_df.columns
        assert feature_df.columns[0].upper() == "F1"
        assert feature_df.columns[-1].upper() == "F230"
    else:
        with pytest.raises(FileNotFoundError) as excinfo:
            load_synthetic_classification()
        assert "data_train_1000.csv" in str(excinfo.value)
        assert "https://www.kaggle.com/competitions/syntheticdata1000" in str(excinfo.value)

"""Unit and integration tests for Data Preprocessing Pipeline across all 5 datasets."""

import os
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset_loader import (
    load_credit_card_customers,
    load_customer_personality,
    load_wine_quality,
    load_online_retail,
    load_synthetic_classification,
    smart_read_table,
    RAW_DATA_DIR,
)
from preprocessing.data_preprocessing import (
    preprocess_credit_card,
    preprocess_customer_personality,
    preprocess_wine_quality,
    preprocess_online_retail,
    preprocess_synthetic_classification,
    preprocess_dataset_by_id,
    ModularPreprocessor,
)


def test_preprocess_credit_card():
    """Verify Credit Card Customers preprocessing."""
    filepath = os.path.join(RAW_DATA_DIR, "credit_card_customers", "BankChurners.csv")
    raw_df = smart_read_table(filepath)
    X, scaled_df, summary = preprocess_credit_card(raw_df)

    assert summary.original_shape == (10127, 23)
    assert X.shape[0] == 10127
    assert X.shape[1] == 32  # 32 one-hot encoded features
    assert "CLIENTNUM" not in scaled_df.columns
    assert "Attrition_Flag" not in scaled_df.columns
    assert not any("naive_bayes" in c.lower() for c in scaled_df.columns)
    assert np.isnan(X).sum() == 0
    assert np.isclose(np.mean(X), 0, atol=1e-1)
    assert summary.is_valid is True


def test_preprocess_customer_personality():
    """Verify Customer Personality preprocessing."""
    filepath = os.path.join(RAW_DATA_DIR, "customer_personality", "marketing_campaign.csv")
    raw_df = smart_read_table(filepath)
    X, scaled_df, summary = preprocess_customer_personality(raw_df)

    assert summary.original_shape == (2240, 29)
    assert X.shape[0] == 2240
    assert "ID" not in scaled_df.columns
    assert "Dt_Customer" not in scaled_df.columns
    assert "Response" not in scaled_df.columns
    assert "Complain" not in scaled_df.columns
    assert "AcceptedCmp1" not in scaled_df.columns
    assert "Customer_For_Days" in scaled_df.columns
    assert np.isnan(X).sum() == 0
    assert summary.missing_values_after == 0
    assert summary.is_valid is True


def test_preprocess_wine_quality():
    """Verify Wine Quality preprocessing."""
    filepath = os.path.join(RAW_DATA_DIR, "wine_quality", "winequality.csv")
    raw_df = smart_read_table(filepath)
    X, scaled_df, summary = preprocess_wine_quality(raw_df)

    assert summary.original_shape == (6497, 14)
    assert X.shape == (6497, 11)  # 11 physicochemical features
    assert "quality" not in scaled_df.columns
    assert "good" not in scaled_df.columns
    assert "color" not in scaled_df.columns
    assert list(scaled_df.columns) == [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol",
    ]
    assert np.isnan(X).sum() == 0
    assert summary.is_valid is True


def test_preprocess_online_retail():
    """Verify Online Retail customer-level RFM preprocessing."""
    filepath = os.path.join(RAW_DATA_DIR, "online_retail", "Online Retail.xlsx")
    raw_df = smart_read_table(filepath)
    X, scaled_df, summary = preprocess_online_retail(raw_df)

    assert summary.original_shape == (541909, 8)
    assert X.shape == (4338, 6)
    expected_features = [
        "Recency", "Frequency", "Monetary",
        "AvgOrderValue", "TotalQuantity", "UniqueProducts",
    ]
    assert list(scaled_df.columns) == expected_features
    assert np.isnan(X).sum() == 0
    assert summary.is_valid is True


def test_preprocess_synthetic_classification():
    """Verify Synthetic Classification 1000 preprocessing."""
    filepath = os.path.join(RAW_DATA_DIR, "synthetic_classification_1000", "data_train_1000.csv")
    raw_df = smart_read_table(filepath)
    X, scaled_df, summary = preprocess_synthetic_classification(raw_df)

    assert summary.original_shape == (1000, 231)
    assert X.shape == (1000, 230)
    assert "Label" not in scaled_df.columns
    assert scaled_df.columns[0] == "F1"
    assert scaled_df.columns[-1] == "F230"
    assert np.isnan(X).sum() == 0
    assert summary.is_valid is True


def test_identical_input_guarantee():
    """Verify that both K-Means and C-Means receive identical feature matrices."""
    filepath = os.path.join(RAW_DATA_DIR, "wine_quality", "winequality.csv")
    raw_df = smart_read_table(filepath)
    X1, _, _ = preprocess_wine_quality(raw_df)
    X2, _, _ = preprocess_wine_quality(raw_df)

    # Pure deterministic output guarantee
    np.testing.assert_array_equal(X1, X2)

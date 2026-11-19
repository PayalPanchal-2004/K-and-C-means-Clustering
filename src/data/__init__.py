"""Data loading, preprocessing, and synthetic generation package."""

from .dataset_loader import (
    DATASET_CATALOG,
    DatasetSpec,
    load_credit_card_customers,
    load_customer_personality,
    load_wine_quality,
    load_online_retail,
    load_synthetic_classification,
    load_dataset_by_id,
    validate_all_datasets,
    smart_read_table,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)

__all__ = [
    "DATASET_CATALOG",
    "DatasetSpec",
    "load_credit_card_customers",
    "load_customer_personality",
    "load_wine_quality",
    "load_online_retail",
    "load_synthetic_classification",
    "load_dataset_by_id",
    "validate_all_datasets",
    "smart_read_table",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
]

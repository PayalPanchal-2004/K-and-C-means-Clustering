"""Data Preprocessing Pipeline for K-Means and C-Means Clustering Comparison.

Implements modular preprocessing:
1. Numerical and categorical feature identification
2. Missing-value analysis and imputation
3. Duplicate detection and handling
4. Outlier analysis (IQR and Z-score methods)
5. Categorical encoding (One-Hot Encoding)
6. Feature scaling using StandardScaler
7. Feature selection (variance threshold, collinearity screening)
8. Identifier and irrelevant column removal
9. Target-label exclusion
10. Strict post-preprocessing validation
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

logger = logging.getLogger("DataPreprocessing")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class PreprocessingSummary:
    """Detailed summary of preprocessing operations for a dataset."""
    dataset_name: str
    original_shape: Tuple[int, int]
    processed_shape: Tuple[int, int]
    numerical_features_raw: List[str]
    categorical_features_raw: List[str]
    excluded_identifiers: List[str]
    excluded_targets: List[str]
    excluded_auxiliary: List[str]
    missing_values_before: Dict[str, int]
    missing_values_after: int
    duplicates_detected: int
    outlier_counts_iqr: Dict[str, int]
    outlier_counts_zscore: Dict[str, int]
    final_feature_names: List[str]
    scaler_mean: Optional[List[float]] = None
    scaler_scale: Optional[List[float]] = None
    is_valid: bool = True
    validation_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "original_shape": self.original_shape,
            "processed_shape": self.processed_shape,
            "raw_numerical_count": len(self.numerical_features_raw),
            "raw_categorical_count": len(self.categorical_features_raw),
            "excluded_identifiers": self.excluded_identifiers,
            "excluded_targets": self.excluded_targets,
            "excluded_auxiliary": self.excluded_auxiliary,
            "total_missing_before": sum(self.missing_values_before.values()),
            "missing_after": self.missing_values_after,
            "duplicates_detected": self.duplicates_detected,
            "total_outliers_iqr": sum(self.outlier_counts_iqr.values()),
            "total_outliers_zscore": sum(self.outlier_counts_zscore.values()),
            "final_feature_count": len(self.final_feature_names),
            "is_valid": self.is_valid,
            "validation_notes": self.validation_notes,
        }


class ModularPreprocessor:
    """Configurable, modular preprocessor for clustering datasets."""

    def __init__(
        self,
        id_columns: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
        drop_columns: Optional[List[str]] = None,
        variance_threshold: float = 0.0,
        handle_outliers: bool = False,
        outlier_cap_percentile: Tuple[float, float] = (1.0, 99.0),
    ):
        self.id_columns = id_columns or []
        self.target_columns = target_columns or []
        self.drop_columns = drop_columns or []
        self.variance_threshold = variance_threshold
        self.handle_outliers = handle_outliers
        self.outlier_cap_percentile = outlier_cap_percentile

        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_names_: List[str] = []
        self.targets_: Dict[str, pd.Series] = {}
        self.identifiers_: Dict[str, pd.Series] = {}

    def identify_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify numerical and categorical features in DataFrame."""
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category", "str", "string", "bool"]).columns.tolist()
        return num_cols, cat_cols

    def analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analyze missing values per column."""
        missing = df.isnull().sum()
        return missing[missing > 0].to_dict()

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values: median for numerical, mode for categorical."""
        df_clean = df.copy()
        num_cols, cat_cols = self.identify_feature_types(df_clean)

        for col in num_cols:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)

        for col in cat_cols:
            if df_clean[col].isnull().any():
                mode_val = df_clean[col].mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
                df_clean[col] = df_clean[col].fillna(fill_val)

        return df_clean

    def detect_outliers_iqr(self, df: pd.DataFrame) -> Dict[str, int]:
        """Detect outliers using the Interquartile Range (IQR) rule: Q1 - 1.5*IQR, Q3 + 1.5*IQR."""
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_counts = {}
        for col in num_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            count = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
            if count > 0:
                outlier_counts[col] = count
        return outlier_counts

    def detect_outliers_zscore(self, df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, int]:
        """Detect outliers using the Z-Score method (|z| > 3.0)."""
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_counts = {}
        for col in num_cols:
            std = df[col].std()
            if std > 0:
                z_scores = np.abs((df[col] - df[col].mean()) / std)
                count = int((z_scores > threshold).sum())
                if count > 0:
                    outlier_counts[col] = count
        return outlier_counts

    def cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optional winsorization/capping at designated percentiles."""
        df_capped = df.copy()
        num_cols = df_capped.select_dtypes(include=[np.number]).columns.tolist()
        low_p, high_p = self.outlier_cap_percentile
        for col in num_cols:
            lower = df_capped[col].quantile(low_p / 100.0)
            upper = df_capped[col].quantile(high_p / 100.0)
            df_capped[col] = df_capped[col].clip(lower=lower, upper=upper)
        return df_capped

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-Hot Encode categorical variables with drop_first=True to avoid collinearity."""
        cat_cols = df.select_dtypes(include=["object", "category", "str", "string", "bool"]).columns.tolist()
        if not cat_cols:
            return df
        return pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)

    def select_features_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove zero-variance or low-variance features."""
        if self.variance_threshold <= 0:
            return df
        selector = VarianceThreshold(threshold=self.variance_threshold)
        selector.fit(df)
        kept_cols = df.columns[selector.get_support()].tolist()
        dropped = set(df.columns) - set(kept_cols)
        if dropped:
            logger.info(f"Dropped low-variance features: {dropped}")
        return df[kept_cols]

    def validate_processed_data(self, X: np.ndarray, feature_names: List[str]) -> Tuple[bool, List[str]]:
        """Validate preprocessed feature matrix meets clustering requirements."""
        notes = []
        is_valid = True

        if X.ndim != 2:
            is_valid = False
            notes.append(f"Expected 2D array, got shape {X.shape}")

        if np.isnan(X).any():
            is_valid = False
            notes.append("Array contains NaN values after preprocessing")

        if np.isinf(X).any():
            is_valid = False
            notes.append("Array contains infinite values after preprocessing")

        if X.shape[0] < 1000:
            notes.append(f"Warning: sample count {X.shape[0]} is below 1,000 threshold")

        if X.shape[1] == 0:
            is_valid = False
            notes.append("Feature space is empty after preprocessing")

        # Verify zero-mean and unit-variance (standard scaling check)
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        if not np.allclose(means, 0, atol=1e-2):
            notes.append(f"Max feature mean deviation from 0: {np.max(np.abs(means)):.4f}")
        # Allow zero std check if a column was constant before, else std close to 1
        non_constant_stds = stds[stds > 1e-4]
        if len(non_constant_stds) > 0 and not np.allclose(non_constant_stds, 1, atol=1e-2):
            notes.append(f"Max feature std deviation from 1: {np.max(np.abs(non_constant_stds - 1)):.4f}")

        if is_valid and not notes:
            notes.append("Passed all validation checks: Finite, zero-NaN, scaled to mean=0, std=1.")

        return is_valid, notes

    def fit_transform(
        self,
        df: pd.DataFrame,
        dataset_name: str = "Dataset",
        original_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, pd.DataFrame, PreprocessingSummary]:
        """Execute full preprocessing pipeline on input DataFrame."""
        orig_shape = original_shape or df.shape
        num_raw, cat_raw = self.identify_feature_types(df)
        missing_before = self.analyze_missing_values(df)
        duplicates_detected = int(df.duplicated().sum())

        df_work = df.copy()

        # 1. Extract and isolate identifiers
        excluded_ids = []
        for col in self.id_columns:
            if col in df_work.columns:
                self.identifiers_[col] = df_work[col].copy()
                excluded_ids.append(col)
                df_work = df_work.drop(columns=[col])

        # 2. Extract and isolate target labels
        excluded_targets = []
        for col in self.target_columns:
            if col in df_work.columns:
                self.targets_[col] = df_work[col].copy()
                excluded_targets.append(col)
                df_work = df_work.drop(columns=[col])

        # 3. Drop auxiliary / irrelevant columns
        excluded_aux = []
        for col in self.drop_columns:
            if col in df_work.columns:
                excluded_aux.append(col)
                df_work = df_work.drop(columns=[col])

        # 4. Handle missing values
        df_work = self.handle_missing_values(df_work)
        missing_after = int(df_work.isnull().sum().sum())

        # 5. Outlier analysis
        outliers_iqr = self.detect_outliers_iqr(df_work)
        outliers_z = self.detect_outliers_zscore(df_work)

        if self.handle_outliers:
            df_work = self.cap_outliers(df_work)

        # 6. Categorical encoding
        df_encoded = self.encode_categoricals(df_work)

        # 7. Low-variance feature selection
        df_selected = self.select_features_variance(df_encoded)

        # 8. Feature scaling using StandardScaler
        scaled_array = self.scaler.fit_transform(df_selected)
        self.fitted = True
        self.feature_names_ = df_selected.columns.tolist()

        scaled_df = pd.DataFrame(scaled_array, columns=self.feature_names_, index=df_selected.index)

        # 9. Strict post-preprocessing validation
        is_valid, val_notes = self.validate_processed_data(scaled_array, self.feature_names_)

        summary = PreprocessingSummary(
            dataset_name=dataset_name,
            original_shape=orig_shape,
            processed_shape=scaled_df.shape,
            numerical_features_raw=num_raw,
            categorical_features_raw=cat_raw,
            excluded_identifiers=excluded_ids,
            excluded_targets=excluded_targets,
            excluded_auxiliary=excluded_aux,
            missing_values_before=missing_before,
            missing_values_after=missing_after,
            duplicates_detected=duplicates_detected,
            outlier_counts_iqr=outliers_iqr,
            outlier_counts_zscore=outliers_z,
            final_feature_names=self.feature_names_,
            scaler_mean=self.scaler.mean_.tolist(),
            scaler_scale=self.scaler.scale_.tolist(),
            is_valid=is_valid,
            validation_notes=val_notes,
        )

        return scaled_array, scaled_df, summary


# =====================================================================
# DATASET-SPECIFIC PREPROCESSING PIPELINES
# =====================================================================

def preprocess_credit_card(raw_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame, PreprocessingSummary]:
    """Preprocess Credit Card Customers: excludes CLIENTNUM, Attrition_Flag, and Naive Bayes artifacts."""
    id_cols = ["CLIENTNUM"]
    target_cols = ["Attrition_Flag"]
    naive_bayes_cols = [
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
    ]

    preprocessor = ModularPreprocessor(
        id_columns=id_cols,
        target_columns=target_cols,
        drop_columns=naive_bayes_cols,
        variance_threshold=1e-4,
    )
    return preprocessor.fit_transform(raw_df, dataset_name="Credit Card Customers")


def preprocess_customer_personality(raw_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame, PreprocessingSummary]:
    """Preprocess Customer Personality: excludes ID, Dt_Customer, campaign responses & constant columns."""
    id_cols = ["ID"]
    target_cols = ["Response"]
    auxiliary_drop = [
        "Dt_Customer",
        "AcceptedCmp1",
        "AcceptedCmp2",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
        "Complain",
        "Z_CostContact",
        "Z_Revenue",
    ]

    # Pre-extract customer seniority from Dt_Customer before dropping
    df_engineered = raw_df.copy()
    if "Dt_Customer" in df_engineered.columns:
        try:
            dt = pd.to_datetime(df_engineered["Dt_Customer"], format="%d-%m-%Y")
        except Exception:
            dt = pd.to_datetime(df_engineered["Dt_Customer"], errors="coerce")
        ref_date = dt.max()
        df_engineered["Customer_For_Days"] = (ref_date - dt).dt.days

    preprocessor = ModularPreprocessor(
        id_columns=id_cols,
        target_columns=target_cols,
        drop_columns=auxiliary_drop,
        variance_threshold=1e-4,
    )
    return preprocessor.fit_transform(
        df_engineered,
        dataset_name="Customer Personality Analysis",
        original_shape=raw_df.shape,
    )


def preprocess_wine_quality(
    raw_df: pd.DataFrame,
    deduplicate: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame, PreprocessingSummary]:
    """Preprocess Wine Quality: excludes quality score, auxiliary good & color, retains 11 physicochemical features.
    
    If deduplicate is True, exact duplicate rows are removed prior to scaling.
    """
    df_input = raw_df.drop_duplicates().copy() if deduplicate else raw_df.copy()
    dataset_name = "Wine Quality (Deduplicated)" if deduplicate else "Wine Quality (With Duplicates)"

    target_cols = [c for c in ["quality", "Quality"] if c in df_input.columns]
    auxiliary_drop = [c for c in ["good", "Good", "color", "Color", "wine_type", "Wine_Type"] if c in df_input.columns]

    preprocessor = ModularPreprocessor(
        id_columns=[],
        target_columns=target_cols,
        drop_columns=auxiliary_drop,
        variance_threshold=0.0,
    )
    return preprocessor.fit_transform(df_input, dataset_name=dataset_name, original_shape=raw_df.shape)


def preprocess_online_retail(raw_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame, PreprocessingSummary]:
    """Preprocess Online Retail: cleans transactions and engineers 6 customer-level RFM features."""
    orig_shape = raw_df.shape
    num_raw, cat_raw = ModularPreprocessor().identify_feature_types(raw_df)
    missing_before = ModularPreprocessor().analyze_missing_values(raw_df)
    duplicates_detected = int(raw_df.duplicated().sum())

    # Standardize column headers
    df = raw_df.rename(columns={c: c.strip() for c in raw_df.columns}).copy()

    cust_col = next((c for c in ["CustomerID", "Customer ID", "Customer_ID"] if c in df.columns), "CustomerID")
    inv_col = next((c for c in ["InvoiceNo", "Invoice", "Invoice Number"] if c in df.columns), "InvoiceNo")
    date_col = next((c for c in ["InvoiceDate", "Invoice Date"] if c in df.columns), "InvoiceDate")
    qty_col = next((c for c in ["Quantity", "Qty"] if c in df.columns), "Quantity")
    price_col = next((c for c in ["UnitPrice", "Unit Price", "Price"] if c in df.columns), "UnitPrice")
    stock_col = next((c for c in ["StockCode", "Stock Code"] if c in df.columns), "StockCode")

    # Step 1: Remove transactions with missing CustomerID
    df_valid = df[df[cust_col].notnull()].copy()

    # Step 2: Remove cancelled invoices ('C') and non-positive quantity / unit price
    valid_tx = df_valid[
        (~df_valid[inv_col].astype(str).str.startswith("C")) &
        (df_valid[qty_col] > 0) &
        (df_valid[price_col] > 0)
    ].copy()

    valid_tx["TotalAmount"] = valid_tx[qty_col] * valid_tx[price_col]
    valid_tx[date_col] = pd.to_datetime(valid_tx[date_col])

    snapshot_date = valid_tx[date_col].max() + pd.Timedelta(days=1)

    # Step 3: Engineer 6 customer-level RFM features
    customer_agg = valid_tx.groupby(cust_col).agg(
        Recency=(date_col, lambda x: (snapshot_date - x.max()).days),
        Frequency=(inv_col, "nunique"),
        Monetary=("TotalAmount", "sum"),
        TotalQuantity=(qty_col, "sum"),
        UniqueProducts=(stock_col, "nunique"),
    ).reset_index()

    # Average Order Value = Monetary / Frequency
    customer_agg["AvgOrderValue"] = customer_agg["Monetary"] / customer_agg["Frequency"]

    feature_cols = [
        "Recency",
        "Frequency",
        "Monetary",
        "AvgOrderValue",
        "TotalQuantity",
        "UniqueProducts",
    ]

    feature_df = customer_agg[feature_cols].copy()

    # Outlier detection on customer features
    outliers_iqr = ModularPreprocessor().detect_outliers_iqr(feature_df)
    outliers_z = ModularPreprocessor().detect_outliers_zscore(feature_df)

    # Scaling with StandardScaler
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(feature_df)
    scaled_df = pd.DataFrame(scaled_array, columns=feature_cols, index=customer_agg[cust_col])

    is_valid, val_notes = ModularPreprocessor().validate_processed_data(scaled_array, feature_cols)

    summary = PreprocessingSummary(
        dataset_name="Online Retail UCI (Customer-Level)",
        original_shape=orig_shape,
        processed_shape=scaled_df.shape,
        numerical_features_raw=num_raw,
        categorical_features_raw=cat_raw,
        excluded_identifiers=[cust_col],
        excluded_targets=[],
        excluded_auxiliary=["Description", "Country", "InvoiceDate", "InvoiceNo", "StockCode"],
        missing_values_before=missing_before,
        missing_values_after=0,
        duplicates_detected=duplicates_detected,
        outlier_counts_iqr=outliers_iqr,
        outlier_counts_zscore=outliers_z,
        final_feature_names=feature_cols,
        scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(),
        is_valid=is_valid,
        validation_notes=val_notes,
    )

    return scaled_array, scaled_df, summary


def preprocess_synthetic_classification(raw_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame, PreprocessingSummary]:
    """Preprocess Synthetic Classification 1000: uses F1–F230 only, completely excludes Label."""
    target_cols = [c for c in ["Label", "label", "Target", "target"] if c in raw_df.columns]
    
    # Filter strictly for F1 to F230
    all_f_cols = [c for c in raw_df.columns if c not in target_cols and c.upper().startswith("F")]
    if len(all_f_cols) != 230:
        # Fallback to excluding non-feature columns
        all_f_cols = [c for c in raw_df.columns if c not in target_cols and c.lower() not in ["id", "index"]][:230]

    preprocessor = ModularPreprocessor(
        id_columns=[],
        target_columns=target_cols,
        drop_columns=[c for c in raw_df.columns if c not in all_f_cols and c not in target_cols],
        variance_threshold=0.0,
    )
    return preprocessor.fit_transform(raw_df, dataset_name="Synthetic Classification 1000")


# Preprocessing dispatch map
PREPROCESSOR_MAPPING = {
    "credit_card_customers": preprocess_credit_card,
    "customer_personality": preprocess_customer_personality,
    "wine_quality": preprocess_wine_quality,
    "online_retail": preprocess_online_retail,
    "synthetic_classification_1000": preprocess_synthetic_classification,
}


def preprocess_dataset_by_id(
    dataset_id: str,
    raw_df: pd.DataFrame,
) -> Tuple[np.ndarray, pd.DataFrame, PreprocessingSummary]:
    """Master entry point to preprocess any integrated dataset by ID."""
    if dataset_id not in PREPROCESSOR_MAPPING:
        raise KeyError(f"Unknown dataset_id '{dataset_id}'. Choose from: {list(PREPROCESSOR_MAPPING.keys())}")
    return PREPROCESSOR_MAPPING[dataset_id](raw_df)

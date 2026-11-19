"""Dataset Loader and Validator for K-Means and C-Means Clustering Comparison.

Provides automated downloading via kagglehub/kaggle API with seamless manual
fallback, intelligent format/encoding detection, data cleaning, RFM customer
aggregation for transactional data, and metadata generation for five specified
Kaggle benchmark datasets.
"""

import os
import shutil
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetLoader")

# Project Directory Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


@dataclass
class DatasetSpec:
    """Metadata specification for a dataset."""
    id: str
    name: str
    description: str
    kaggle_url: str
    kaggle_handle: str
    is_competition: bool
    expected_files: List[str]
    target_column: Optional[str]
    id_columns: List[str] = field(default_factory=list)
    auxiliary_drop_columns: List[str] = field(default_factory=list)
    min_required_records: int = 1000


DATASET_CATALOG: Dict[str, DatasetSpec] = {
    "credit_card_customers": DatasetSpec(
        id="credit_card_customers",
        name="Credit Card Customers (BankChurners)",
        description=(
            "Consumer credit card portfolio dataset containing customer demographics, "
            "credit limits, revolving balances, transaction frequencies, and churn status."
        ),
        kaggle_url="https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers",
        kaggle_handle="sakshigoyal7/credit-card-customers",
        is_competition=False,
        expected_files=["BankChurners.csv"],
        target_column="Attrition_Flag",
        id_columns=["CLIENTNUM"],
        auxiliary_drop_columns=[
            "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
            "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
        ],
        min_required_records=10000,
    ),
    "customer_personality": DatasetSpec(
        id="customer_personality",
        name="Customer Personality Analysis",
        description=(
            "Detailed analysis of an enterprise company's ideal customers, encompassing "
            "customer demographics, spending behavior across categories, campaign responses, "
            "and purchasing channels."
        ),
        kaggle_url="https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis",
        kaggle_handle="imakash3011/customer-personality-analysis",
        is_competition=False,
        expected_files=["marketing_campaign.csv"],
        target_column="Response",
        id_columns=["ID"],
        auxiliary_drop_columns=["Z_CostContact", "Z_Revenue"],  # Constant zero-variance columns
        min_required_records=2000,
    ),
    "wine_quality": DatasetSpec(
        id="wine_quality",
        name="Wine Quality (Red & White Combined)",
        description=(
            "Physicochemical properties of red and white variants of the Portuguese "
            "'Vinho Verde' wine (acidity, residual sugar, chlorides, sulfur dioxide, pH, alcohol) "
            "and sensory quality scores."
        ),
        kaggle_url="https://www.kaggle.com/datasets/aleixdorca/wine-quality",
        kaggle_handle="aleixdorca/wine-quality",
        is_competition=False,
        expected_files=["winequality-red.csv", "winequality-white.csv", "WineQuality.csv"],
        target_column="quality",
        id_columns=[],
        auxiliary_drop_columns=[],
        min_required_records=6000,
    ),
    "online_retail": DatasetSpec(
        id="online_retail",
        name="Online Retail UCI (Customer RFM Segmentation)",
        description=(
            "Transnational dataset containing all transactions occurring between 01/12/2010 "
            "and 09/12/2011 for a UK-based non-store online retail platform. Aggregated into "
            "RFM (Recency, Frequency, Monetary) metrics per unique customer."
        ),
        kaggle_url="https://www.kaggle.com/datasets/jihyeseo/online-retail-data-set-from-uci-ml-repo",
        kaggle_handle="jihyeseo/online-retail-data-set-from-uci-ml-repo",
        is_competition=False,
        expected_files=["Online Retail.xlsx", "OnlineRetail.csv", "online_retail.csv", "Online_Retail.xlsx"],
        target_column=None,
        id_columns=["CustomerID"],
        auxiliary_drop_columns=[],
        min_required_records=4000,  # Unique customers threshold
    ),
    "synthetic_classification_1000": DatasetSpec(
        id="synthetic_classification_1000",
        name="Synthetic Classification 1000",
        description=(
            "Kaggle synthetic classification challenge containing exactly 1,000 training "
            "cases and 231 columns (230 continuous features F1–F230 plus the Label class target)."
        ),
        kaggle_url="https://www.kaggle.com/competitions/syntheticdata1000",
        kaggle_handle="syntheticdata1000",
        is_competition=True,
        expected_files=["data_train_1000.csv", "train.csv"],
        target_column="Label",
        id_columns=["id", "ID"],
        auxiliary_drop_columns=[],
        min_required_records=1000,
    ),
}


def smart_read_table(filepath: str, default_sep: Optional[str] = None) -> pd.DataFrame:
    """Read tabular data with automated encoding detection and delimiter sniffing."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(filepath, engine="openpyxl" if ext == ".xlsx" else None)

    encodings = ["utf-8", "utf-8-sig", "latin1", "iso-8859-1", "cp1252"]

    for enc in encodings:
        try:
            if default_sep is not None:
                return pd.read_csv(filepath, sep=default_sep, encoding=enc)

            # Peek first 20 lines to sniff delimiter
            with open(filepath, "r", encoding=enc) as f:
                sample_lines = [f.readline() for _ in range(15)]
            sample = "".join([line for line in sample_lines if line])

            # Test tab, semicolon, comma
            tab_count = sample.count("\t")
            semi_count = sample.count(";")
            comma_count = sample.count(",")

            if tab_count > semi_count and tab_count > comma_count:
                chosen_sep = "\t"
            elif semi_count > comma_count:
                chosen_sep = ";"
            else:
                chosen_sep = ","

            return pd.read_csv(filepath, sep=chosen_sep, encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    raise ValueError(f"Unable to parse {filepath} with supported encodings and delimiters.")


def download_dataset(spec: DatasetSpec, target_dir: Optional[str] = None) -> str:
    """Attempt automated download via kagglehub, fallback to checking local directory."""
    target_dir = target_dir or os.path.join(RAW_DATA_DIR, spec.id)
    os.makedirs(target_dir, exist_ok=True)

    # Check if expected files are already present
    existing_files = os.listdir(target_dir)
    for expected in spec.expected_files:
        if any(expected.lower() == f.lower() for f in existing_files):
            logger.info(f"[{spec.name}] Found existing file in {target_dir}")
            return target_dir

    # Also check if files are sitting directly in data/raw/
    raw_root_files = os.listdir(RAW_DATA_DIR)
    for expected in spec.expected_files:
        matches = [f for f in raw_root_files if f.lower() == expected.lower()]
        if matches:
            src_path = os.path.join(RAW_DATA_DIR, matches[0])
            dest_path = os.path.join(target_dir, matches[0])
            shutil.copy2(src_path, dest_path)
            logger.info(f"[{spec.name}] Copied {matches[0]} from raw root to {target_dir}")
            return target_dir

    # Attempt automated kagglehub download
    try:
        import kagglehub
        logger.info(f"[{spec.name}] Attempting automated download via kagglehub ({spec.kaggle_handle})...")
        if spec.is_competition:
            download_path = kagglehub.competition_download(spec.kaggle_handle)
        else:
            download_path = kagglehub.dataset_download(spec.kaggle_handle)

        logger.info(f"[{spec.name}] kagglehub download finished at: {download_path}")

        # Copy all contents from kagglehub cache into target_dir
        if os.path.isdir(download_path):
            for item in os.listdir(download_path):
                s = os.path.join(download_path, item)
                d = os.path.join(target_dir, item)
                if os.path.isfile(s):
                    shutil.copy2(s, d)
                elif os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
        return target_dir
    except Exception as e:
        logger.warning(
            f"[{spec.name}] Automated download via kagglehub unavailable: {e}\n"
            f"--> MANUAL DOWNLOAD REQUIRED:\n"
            f"    1. Open: {spec.kaggle_url}\n"
            f"    2. Download the dataset archive.\n"
            f"    3. Extract and place {spec.expected_files} into:\n"
            f"       {os.path.abspath(target_dir)}/"
        )
        return target_dir


# =====================================================================
# DATASET-SPECIFIC INGESTION & PROCESSING
# =====================================================================

def load_credit_card_customers(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load and process Credit Card Customers dataset."""
    spec = DATASET_CATALOG["credit_card_customers"]
    target_dir = download_dataset(spec, data_dir)

    filepath = None
    for f in os.listdir(target_dir):
        if "bankchurners" in f.lower() and f.endswith(".csv"):
            filepath = os.path.join(target_dir, f)
            break

    if not filepath:
        raise FileNotFoundError(
            f"[{spec.name}] BankChurners.csv not found in {target_dir}. "
            f"Please download from {spec.kaggle_url} and place it in {target_dir}/"
        )

    raw_df = smart_read_table(filepath)
    orig_rows, orig_cols = raw_df.shape

    # Identify columns to drop (IDs, target, naive bayes artifact columns)
    drop_cols = []
    for col in spec.id_columns:
        if col in raw_df.columns:
            drop_cols.append(col)

    target_series = None
    if spec.target_column and spec.target_column in raw_df.columns:
        target_series = raw_df[spec.target_column]
        drop_cols.append(spec.target_column)

    # Drop Naive Bayes prediction artifacts
    for col in raw_df.columns:
        if "naive_bayes_classifier" in col.lower():
            drop_cols.append(col)

    feature_df = raw_df.drop(columns=drop_cols, errors="ignore").copy()

    # Encode categorical features for clustering
    categorical_cols = feature_df.select_dtypes(include=["object", "category", "str", "string"]).columns.tolist()
    if categorical_cols:
        feature_df = pd.get_dummies(feature_df, columns=categorical_cols, drop_first=True)

    # Impute missing values if any
    num_missing = feature_df.isnull().sum().sum()
    if num_missing > 0:
        feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

    clean_path = os.path.join(PROCESSED_DATA_DIR, f"{spec.id}_clean.csv")
    feature_df.to_csv(clean_path, index=False)

    summary = {
        "dataset_id": spec.id,
        "name": spec.name,
        "description": spec.description,
        "kaggle_url": spec.kaggle_url,
        "raw_rows": orig_rows,
        "raw_cols": orig_cols,
        "processed_rows": len(feature_df),
        "clustering_features": feature_df.shape[1],
        "excluded_identifiers": [c for c in spec.id_columns if c in raw_df.columns],
        "excluded_target": spec.target_column,
        "missing_values_raw": int(raw_df.isnull().sum().sum()),
        "duplicate_rows_raw": int(raw_df.duplicated().sum()),
        "target_distribution": target_series.value_counts().to_dict() if target_series is not None else None,
        "clean_csv_path": clean_path,
    }
    return raw_df, feature_df, summary


def load_customer_personality(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load and process Customer Personality Analysis dataset."""
    spec = DATASET_CATALOG["customer_personality"]
    target_dir = download_dataset(spec, data_dir)

    filepath = None
    for f in os.listdir(target_dir):
        if "marketing_campaign" in f.lower() and f.endswith(".csv"):
            filepath = os.path.join(target_dir, f)
            break

    if not filepath:
        raise FileNotFoundError(
            f"[{spec.name}] marketing_campaign.csv not found in {target_dir}. "
            f"Please download from {spec.kaggle_url} and place it in {target_dir}/"
        )

    raw_df = smart_read_table(filepath)
    orig_rows, orig_cols = raw_df.shape

    drop_cols = []
    for col in spec.id_columns:
        if col in raw_df.columns:
            drop_cols.append(col)

    target_series = None
    if spec.target_column and spec.target_column in raw_df.columns:
        target_series = raw_df[spec.target_column]
        drop_cols.append(spec.target_column)

    for col in spec.auxiliary_drop_columns:
        if col in raw_df.columns:
            drop_cols.append(col)

    # Handle Dt_Customer (date) by extracting customer seniority in days
    feature_df = raw_df.drop(columns=drop_cols, errors="ignore").copy()
    if "Dt_Customer" in feature_df.columns:
        try:
            dt = pd.to_datetime(feature_df["Dt_Customer"], format="%d-%m-%Y")
        except Exception:
            dt = pd.to_datetime(feature_df["Dt_Customer"], errors="coerce")
        ref_date = dt.max()
        feature_df["Customer_For_Days"] = (ref_date - dt).dt.days
        feature_df = feature_df.drop(columns=["Dt_Customer"])

    # Encode categorical columns
    cat_cols = feature_df.select_dtypes(include=["object", "category", "str", "string"]).columns.tolist()
    if cat_cols:
        feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=True)

    # Impute missing values (e.g. Income column has 24 missing values in this dataset)
    raw_missing = int(raw_df.isnull().sum().sum())
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

    clean_path = os.path.join(PROCESSED_DATA_DIR, f"{spec.id}_clean.csv")
    feature_df.to_csv(clean_path, index=False)

    summary = {
        "dataset_id": spec.id,
        "name": spec.name,
        "description": spec.description,
        "kaggle_url": spec.kaggle_url,
        "raw_rows": orig_rows,
        "raw_cols": orig_cols,
        "processed_rows": len(feature_df),
        "clustering_features": feature_df.shape[1],
        "excluded_identifiers": [c for c in spec.id_columns if c in raw_df.columns],
        "excluded_target": spec.target_column,
        "missing_values_raw": raw_missing,
        "duplicate_rows_raw": int(raw_df.duplicated().sum()),
        "target_distribution": target_series.value_counts().to_dict() if target_series is not None else None,
        "clean_csv_path": clean_path,
    }
    return raw_df, feature_df, summary


def load_wine_quality(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load and process Wine Quality dataset (verifying combined red & white count)."""
    spec = DATASET_CATALOG["wine_quality"]
    target_dir = download_dataset(spec, data_dir)

    # Check for combined file or separate red/white files
    combined_file = None
    red_file = None
    white_file = None

    for f in os.listdir(target_dir):
        f_low = f.lower()
        if "winequality.csv" in f_low or "wine_quality.csv" in f_low:
            combined_file = os.path.join(target_dir, f)
        elif "winequality-red" in f_low or "wine_quality_red" in f_low:
            red_file = os.path.join(target_dir, f)
        elif "winequality-white" in f_low or "wine_quality_white" in f_low:
            white_file = os.path.join(target_dir, f)

    if combined_file:
        raw_df = smart_read_table(combined_file)
    elif red_file and white_file:
        red_df = smart_read_table(red_file)
        white_df = smart_read_table(white_file)
        red_df["wine_type"] = "red"
        white_df["wine_type"] = "white"
        raw_df = pd.concat([red_df, white_df], ignore_index=True)
    elif red_file or white_file:
        single_file = red_file or white_file
        raw_df = smart_read_table(single_file)
    else:
        raise FileNotFoundError(
            f"[{spec.name}] Wine quality files not found in {target_dir}. "
            f"Please download from {spec.kaggle_url} and place winequality-red.csv & "
            f"winequality-white.csv (or WineQuality.csv) into {target_dir}/"
        )

    orig_rows, orig_cols = raw_df.shape

    # Exclude quality target, good, and optional wine_type / color from clustering features
    drop_cols = []
    target_series = None
    for target_candidate in ["quality", "Quality"]:
        if target_candidate in raw_df.columns:
            target_series = raw_df[target_candidate]
            drop_cols.append(target_candidate)

    for aux in ["good", "Good", "color", "Color", "wine_type", "Wine_Type"]:
        if aux in raw_df.columns:
            drop_cols.append(aux)

    feature_df = raw_df.drop(columns=drop_cols, errors="ignore").copy()
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

    clean_path = os.path.join(PROCESSED_DATA_DIR, f"{spec.id}_clean.csv")
    feature_df.to_csv(clean_path, index=False)

    summary = {
        "dataset_id": spec.id,
        "name": spec.name,
        "description": spec.description,
        "kaggle_url": spec.kaggle_url,
        "raw_rows": orig_rows,
        "raw_cols": orig_cols,
        "combined_wine_count_verified": orig_rows >= 6000,
        "processed_rows": len(feature_df),
        "clustering_features": feature_df.shape[1],
        "excluded_identifiers": [],
        "excluded_target": "quality",
        "missing_values_raw": int(raw_df.isnull().sum().sum()),
        "duplicate_rows_raw": int(raw_df.duplicated().sum()),
        "target_distribution": target_series.value_counts().to_dict() if target_series is not None else None,
        "clean_csv_path": clean_path,
    }
    return raw_df, feature_df, summary


def load_online_retail(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load Online Retail dataset and perform customer RFM aggregation."""
    spec = DATASET_CATALOG["online_retail"]
    target_dir = download_dataset(spec, data_dir)

    filepath = None
    for f in os.listdir(target_dir):
        f_low = f.lower()
        if "online" in f_low and "retail" in f_low and (f.endswith(".xlsx") or f.endswith(".csv")):
            filepath = os.path.join(target_dir, f)
            break

    if not filepath:
        raise FileNotFoundError(
            f"[{spec.name}] Online Retail file not found in {target_dir}. "
            f"Please download from {spec.kaggle_url} and place 'Online Retail.xlsx' or "
            f"'OnlineRetail.csv' in {target_dir}/"
        )

    raw_df = smart_read_table(filepath)
    raw_transaction_count, orig_cols = raw_df.shape

    # Standardize column names
    col_mapping = {c: c.strip() for c in raw_df.columns}
    raw_df = raw_df.rename(columns=col_mapping)

    # Customer ID column detection
    cust_col = next((c for c in ["CustomerID", "Customer ID", "Customer_ID"] if c in raw_df.columns), None)
    inv_col = next((c for c in ["InvoiceNo", "Invoice", "Invoice Number"] if c in raw_df.columns), None)
    date_col = next((c for c in ["InvoiceDate", "Invoice Date"] if c in raw_df.columns), None)
    qty_col = next((c for c in ["Quantity", "Qty"] if c in raw_df.columns), None)
    price_col = next((c for c in ["UnitPrice", "Unit Price", "Price"] if c in raw_df.columns), None)

    if not (cust_col and inv_col and date_col and qty_col and price_col):
        raise ValueError(f"Online Retail dataset missing required transactional columns: {raw_df.columns.tolist()}")

    # Count transactions with valid CustomerID
    valid_customer_transactions = raw_df[raw_df[cust_col].notnull()].copy()

    # Filter out cancellations (InvoiceNo starting with 'C') and non-positive quantities/prices
    valid_tx = valid_customer_transactions[
        (~valid_customer_transactions[inv_col].astype(str).str.startswith("C")) &
        (valid_customer_transactions[qty_col] > 0) &
        (valid_customer_transactions[price_col] > 0)
    ].copy()

    valid_tx["TotalAmount"] = valid_tx[qty_col] * valid_tx[price_col]
    valid_tx[date_col] = pd.to_datetime(valid_tx[date_col])

    # Reference date for recency calculation (1 day after latest transaction)
    snapshot_date = valid_tx[date_col].max() + pd.Timedelta(days=1)

    # RFM Aggregation by CustomerID
    rfm_df = valid_tx.groupby(cust_col).agg({
        date_col: lambda x: (snapshot_date - x.max()).days,  # Recency
        inv_col: "nunique",                                  # Frequency
        "TotalAmount": "sum",                                # Monetary
    }).reset_index()

    rfm_df.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
    unique_customer_count = len(rfm_df)

    # Exclude CustomerID from clustering features
    feature_df = rfm_df[["Recency", "Frequency", "Monetary"]].copy()

    clean_path = os.path.join(PROCESSED_DATA_DIR, f"{spec.id}_clean.csv")
    feature_df.to_csv(clean_path, index=False)

    summary = {
        "dataset_id": spec.id,
        "name": spec.name,
        "description": spec.description,
        "kaggle_url": spec.kaggle_url,
        "original_transaction_count": raw_transaction_count,
        "raw_rows": raw_transaction_count,
        "raw_cols": orig_cols,
        "unique_customer_count_aggregated": unique_customer_count,
        "processed_rows": unique_customer_count,
        "clustering_features": feature_df.shape[1],
        "excluded_identifiers": ["CustomerID"],
        "excluded_target": None,
        "missing_values_raw": int(raw_df.isnull().sum().sum()),
        "duplicate_rows_raw": int(raw_df.duplicated().sum()),
        "rfm_columns": ["Recency", "Frequency", "Monetary"],
        "clean_csv_path": clean_path,
    }
    return raw_df, feature_df, summary


def load_synthetic_classification(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load Synthetic Classification 1000 dataset (data_train_1000.csv, F1–F230 only, excluding Label)."""
    spec = DATASET_CATALOG["synthetic_classification_1000"]
    target_dir = download_dataset(spec, data_dir)

    filepath = None
    # Prioritize 'data_train_1000.csv' and match variants
    search_names = ["data_train_1000.csv", "data_train_1000", "train.csv", "syntheticdata1000.csv"]
    for candidate in search_names:
        for f in os.listdir(target_dir):
            if f.lower() == candidate.lower() or ("data_train_1000" in f.lower() and f.endswith(".csv")):
                filepath = os.path.join(target_dir, f)
                break
        if filepath:
            break

    # Also check if user placed it in raw root or Downloads
    if not filepath:
        for f in os.listdir(RAW_DATA_DIR):
            if "data_train_1000" in f.lower() and f.endswith(".csv"):
                dest = os.path.join(target_dir, "data_train_1000.csv")
                shutil.copy2(os.path.join(RAW_DATA_DIR, f), dest)
                filepath = dest
                break

    if not filepath:
        raise FileNotFoundError(
            f"[{spec.name}] Training file 'data_train_1000.csv' not found in {target_dir}.\n"
            f"Please download data_train_1000.csv from {spec.kaggle_url}/data and place it at:\n"
            f"  {os.path.join(target_dir, 'data_train_1000.csv')}\n"
            f"(Phase 1 remains INCOMPLETE until data_train_1000.csv is verified on disk)."
        )

    raw_df = smart_read_table(filepath)
    orig_rows, orig_cols = raw_df.shape

    # STRICT REQUIREMENT: Confirm exactly 1,000 records
    if orig_rows != 1000:
        raise ValueError(
            f"[{spec.name}] Expected exactly 1,000 records, but found {orig_rows}"
        )

    # STRICT REQUIREMENT: Confirm exactly 231 columns (230 features + Label)
    if orig_cols != 231:
        logger.warning(
            f"[{spec.name}] Expected 231 columns, found {orig_cols}."
        )

    # STRICT REQUIREMENT: Exclude Label completely from clustering and preprocessing
    target_col_name = next((c for c in ["Label", "label", "Target", "target"] if c in raw_df.columns), None)
    target_series = raw_df[target_col_name] if target_col_name else None

    # STRICT REQUIREMENT: Use only F1–F230 for clustering
    all_feature_candidates = [c for c in raw_df.columns if c != target_col_name]
    f_prefixed = [c for c in all_feature_candidates if c.upper().startswith("F")]
    if len(f_prefixed) == 230:
        feature_cols = f_prefixed
    else:
        # Filter out any lingering id / index columns to get the 230 continuous features
        feature_cols = [c for c in all_feature_candidates if c.lower() not in ["id", "unnamed: 0", "index"]][:230]

    feature_df = raw_df[feature_cols].copy()
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

    clean_path = os.path.join(PROCESSED_DATA_DIR, f"{spec.id}_clean.csv")
    feature_df.to_csv(clean_path, index=False)

    summary = {
        "dataset_id": spec.id,
        "name": spec.name,
        "description": spec.description,
        "kaggle_url": spec.kaggle_url,
        "verified_filename": os.path.basename(filepath),
        "raw_rows": orig_rows,
        "raw_cols": orig_cols,
        "processed_rows": len(feature_df),
        "clustering_features": feature_df.shape[1],
        "feature_range": f"{feature_cols[0]} to {feature_cols[-1]} (230 features)",
        "excluded_identifiers": [c for c in raw_df.columns if c not in feature_cols and c != target_col_name],
        "excluded_target": target_col_name,
        "missing_values_raw": int(raw_df.isnull().sum().sum()),
        "duplicate_rows_raw": int(raw_df.duplicated().sum()),
        "target_distribution": target_series.value_counts().to_dict() if target_series is not None else None,
        "clean_csv_path": clean_path,
    }
    return raw_df, feature_df, summary


LOADER_MAPPING = {
    "credit_card_customers": load_credit_card_customers,
    "customer_personality": load_customer_personality,
    "wine_quality": load_wine_quality,
    "online_retail": load_online_retail,
    "synthetic_classification_1000": load_synthetic_classification,
}


def load_dataset_by_id(dataset_id: str, data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Master entry point to load any of the 5 datasets by identifier."""
    if dataset_id not in LOADER_MAPPING:
        raise KeyError(f"Unknown dataset_id '{dataset_id}'. Choose from: {list(LOADER_MAPPING.keys())}")
    return LOADER_MAPPING[dataset_id](data_dir)


def validate_all_datasets() -> Dict[str, Dict[str, Any]]:
    """Scan and validate all 5 datasets, returning status and metadata summaries."""
    results = {}
    for dataset_id, loader_fn in LOADER_MAPPING.items():
        spec = DATASET_CATALOG[dataset_id]
        try:
            raw_df, feature_df, summary = loader_fn()
            results[dataset_id] = {
                "status": "LOADED_SUCCESSFULLY",
                "summary": summary,
                "error": None,
            }
            logger.info(
                f"✅ {spec.name}: {summary['raw_rows']} raw records -> "
                f"{summary['processed_rows']} clustering records ({summary['clustering_features']} features)"
            )
        except Exception as e:
            results[dataset_id] = {
                "status": "MISSING_OR_ERROR",
                "summary": None,
                "error": str(e),
                "manual_instructions": {
                    "dataset_name": spec.name,
                    "kaggle_url": spec.kaggle_url,
                    "expected_files": spec.expected_files,
                    "destination_folder": os.path.abspath(os.path.join(RAW_DATA_DIR, spec.id)),
                },
            }
            logger.warning(f"⚠️ {spec.name}: {e}")
    return results


if __name__ == "__main__":
    print("=== DATASET VALIDATION RUN ===")
    validation_report = validate_all_datasets()
    for d_id, rep in validation_report.items():
        print(f"\n[{d_id}] Status: {rep['status']}")
        if rep['status'] == 'LOADED_SUCCESSFULLY':
            print(f"  Raw Rows: {rep['summary']['raw_rows']}, Cols: {rep['summary']['raw_cols']}")
            print(f"  Clustering Rows: {rep['summary']['processed_rows']}, Features: {rep['summary']['clustering_features']}")
        else:
            print(f"  Error: {rep['error']}")
            print(f"  Download URL: {rep['manual_instructions']['kaggle_url']}")
            print(f"  Destination: {rep['manual_instructions']['destination_folder']}")

# Comparison of C-Means Clustering and K-Means Clustering Algorithm: Their Suitability and Applicability

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/Academic-Project-green.svg)]()

---

## 📌 Project Overview

Clustering is a foundational unsupervised machine learning technique aimed at partitioning unlabeled data into cohesive clusters such that items within the same cluster exhibit high mutual similarity while remaining distinct from items in other clusters.

Among partitional clustering algorithms, **K-Means** (MacQueen, 1967) and **Fuzzy C-Means / FCM** (Bezdek, 1984) represent two contrasting paradigms:
- **K-Means**: A **hard (crisp) partitioning** algorithm where each data point strictly belongs to exactly one cluster ($\{0, 1\}$ membership).
- **Fuzzy C-Means (C-Means)**: A **soft (fuzzy) partitioning** algorithm where each data point belongs to multiple clusters simultaneously with real-valued membership degrees $u_{ij} \in [0, 1]$ satisfying $\sum_{j=1}^c u_{ij} = 1$.

While K-Means is computationally efficient and scales linearly, it struggles with non-spherical clusters, overlapping data distributions, and outlier sensitivity. Conversely, C-Means accommodates ambiguity and boundary overlap via fractional membership, but incurs higher computational complexity and sensitivity to the fuzzifier parameter ($m$ or $\alpha$) and centroid initialization.

This project delivers a **rigorous, reproducible, and interactive empirical comparison** of K-Means and Fuzzy C-Means algorithms across standard benchmark datasets and custom synthetic conditions (overlapping, noisy, and high-dimensional), complete with a professional Streamlit dashboard.

---

## 🎓 Academic Credentials

- **Project Title:** Comparison of C-Means Clustering and K-Means Clustering Algorithm: Their Suitability and Applicability
- **Program:** Master of Computer Applications (MCA) in Artificial Intelligence & Data Science
- **Candidate:** Payal (University Roll No: 1103739)
- **Supervisor:** Dr. Sushil Dmiri
- **Institution:** Department of Computer Applications, Graphic Era (Deemed to be University), Dehradun, Uttarakhand, India
- **Academic Year:** 2026

---

## 🎯 Project Objectives

In alignment with Chapter 3 of the project synopsis:

1. **Fundamental Study:** Analyze the mathematical formulations and iterative convergence mechanisms of K-Means (crisp) and C-Means (fuzzy).
2. **Algorithm Implementation:** Implement both algorithms with transparent tracking of objective function trajectories, centroid paths, and membership values.
3. **Performance Benchmarking:** Compare both algorithms across clustering quality (SSE, RMSE, Silhouette, Davies-Bouldin), computational time ($ms$), and iterations to convergence.
4. **Suitability & Boundary Analysis:** Systematically evaluate performance across complex data landscapes:
   - Overlapping clusters (varying separation factor)
   - High-dimensional feature spaces
   - Noisy data with synthetic outliers
5. **Guideline & Recommendation System:** Provide actionable criteria and an automated recommendation engine to guide practitioners on algorithm selection based on dataset properties.

---

## 📐 Mathematical Formulations

### 1. K-Means Clustering (MacQueen, 1967)

K-Means minimizes the within-cluster Sum of Squared Errors (SSE / Inertia):

$$J_{K\text{-Means}} = \sum_{j=1}^{k} \sum_{x_i \in C_j} \| x_i - c_j \|^2$$

- **Partition Matrix Assignment:**
  $$w_{ij} = \begin{cases} 1 & \text{if } j = \arg\min_l \| x_i - c_l \|^2 \\ 0 & \text{otherwise} \end{cases}$$
- **Centroid Update Rule:**
  $$c_j = \frac{\sum_{i=1}^n w_{ij} x_i}{\sum_{i=1}^n w_{ij}}$$

---

### 2. Fuzzy C-Means (FCM) Clustering (Bezdek, 1984)

FCM minimizes the generalized fuzzy objective function:

$$J_m(U, V) = \sum_{i=1}^{n} \sum_{j=1}^{c} u_{ij}^m \| x_i - v_j \|^2$$

Subject to:
$$\sum_{j=1}^{c} u_{ij} = 1 \quad \forall i \in \{1, \dots, n\}, \quad u_{ij} \in [0, 1]$$

Where:
- $n$: Number of data samples
- $c$: Number of clusters ($k$)
- $m$: Fuzzifier weighting exponent ($m > 1$, typically $m=2.0$, referred to as $\alpha$ in synopsis)
- $u_{ij}$: Membership degree of point $x_i$ in cluster $j$
- $v_j$: Cluster center of cluster $j$

- **Membership Update Rule:**
  $$u_{ij} = \frac{1}{\sum_{l=1}^{c} \left( \frac{\| x_i - v_j \|}{\| x_i - v_l \|} \right)^{\frac{2}{m - 1}}}$$
- **Centroid Update Rule:**
  $$v_j = \frac{\sum_{i=1}^{n} u_{ij}^m x_i}{\sum_{i=1}^{n} u_{ij}^m}$$
- **Defuzzification (Crisp Labeling):**
  $$\hat{y}_i = \arg\max_{j \in \{1, \dots, c\}} u_{ij}$$

---

### 3. Evaluation Metrics

1. **Sum of Squared Errors (SSE):**
   $$\text{SSE} = \sum_{i=1}^n \| x_i - c_{\hat{y}_i} \|^2$$
2. **Root Mean Square Error (RMSE) to Centroids (Synopsis Chapter 5):**
   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n \| x_i - c_{\hat{y}_i} \|^2}$$
   *(And fuzzy weighted RMSE: $\sqrt{\frac{1}{n \cdot d} \sum_{i=1}^n \sum_{j=1}^c u_{ij}^m \|x_i - v_j\|^2}$)*
3. **Silhouette Coefficient:** Measures cohesion vs separation ($s \in [-1, 1]$).
4. **Davies-Bouldin Index (DBI):** Ratio of within-cluster distance to between-cluster distance (lower is better).
5. **Calinski-Harabasz Index (CHI):** Ratio of between-cluster variance to within-cluster variance (higher is better).
6. **Fuzzy Partition Coefficient (FPC):** $FPC = \frac{1}{n} \sum_{i=1}^n \sum_{j=1}^c u_{ij}^2 \in [\frac{1}{c}, 1]$.
7. **External Metrics (with ground truth labels):** Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and Hungarian Optimal Matching Accuracy.

---

## 🏗️ Project Architecture & Directory Layout

```
.
├── app.py                          # Streamlit Main Application Entry Point
├── requirements.txt                # Project Python dependencies
├── .gitignore                      # Git ignore file
├── README.md                       # Comprehensive Project Documentation
│
├── pages/                          # Multi-Page Streamlit Frontend
│   ├── 1_📖_Theoretical_Overview.py # Mathematical comparison & theoretical guide
│   ├── 2_📊_Dataset_Lab.py          # Benchmark selection, synthetic generator, custom upload
│   ├── 3_🔬_Clustering_Execution.py # Interactive model execution & real-time convergence
│   ├── 4_📈_Comparative_Analysis.py# Comprehensive comparative benchmarks & visual analytics
│   ├── 5_🧪_Robustness_&_Sensitivity.py # Overlap, noise, and parameter stress tests
│   └── 6_🎯_Recommendation_Engine.py # Automated algorithm selection & exportable report
│
├── src/                            # Modular Machine Learning Core
│   ├── __init__.py
│   ├── models/                     # Clustering Algorithms
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract base clusterer with uniform API
│   │   ├── kmeans_model.py         # K-Means model with diagnostics & trajectory tracking
│   │   └── fcm_model.py            # Fuzzy C-Means (NumPy vectorised & skfuzzy engine)
│   │
│   ├── data/                       # Data Pipeline
│   │   ├── __init__.py
│   │   ├── datasets.py             # Benchmark loaders (Iris, Wine, Breast Cancer, Mall)
│   │   ├── generator.py            # Synthetic generators (overlapping, noisy, high-dim)
│   │   └── preprocessor.py         # Feature scaling, missing values, PCA/t-SNE projection
│   │
│   ├── evaluation/                 # Metrics & Benchmarking
│   │   ├── __init__.py
│   │   ├── metrics.py              # SSE, RMSE, Silhouette, DBI, CHI, FPC, ARI, NMI
│   │   ├── comparator.py           # Unified comparison engine & benchmark execution
│   │   └── sensitivity.py          # Parametric sweeps (noise %, overlap std, fuzzifier m)
│   │
│   └── visualization/              # Plotly & Matplotlib Plotting Components
│       ├── __init__.py
│       ├── cluster_plots.py        # 2D/3D scatter plots with interactive centroids
│       ├── fuzzy_plots.py          # Membership heatmaps, fuzzy boundary surfaces
│       └── metric_plots.py         # Radar charts, convergence curves, sensitivity plots
│
├── config/                         # Application Configuration
│   ├── __init__.py
│   └── settings.py                 # Default parameters, color palettes, dataset metadata
│
├── data/                           # Dataset Storage
│   ├── raw/                        # Raw local benchmark datasets
│   └── processed/                  # Cached preprocessed datasets
│
└── tests/                          # Automated Verification
    ├── __init__.py
    ├── test_kmeans.py              # Unit tests for K-Means implementation
    ├── test_fcm.py                 # Unit tests for Fuzzy C-Means implementation
    └── test_metrics.py             # Unit tests for evaluation metrics (SSE, RMSE, etc.)
```

---

## ⚡ Technical Requirements & Libraries

| Category | Library | Purpose |
|---|---|---|
| **Language** | `Python >= 3.10` | Core programming language |
| **Data Processing** | `NumPy`, `Pandas`, `SciPy` | Array operations, matrix manipulation, optimization |
| **Machine Learning** | `scikit-learn` | K-Means, dimensionality reduction (PCA/t-SNE), standard metrics |
| **Fuzzy Logic** | `scikit-fuzzy` | Native Fuzzy C-Means clustering engine & fuzzy tools |
| **Interactive UI** | `Streamlit` | Multi-page interactive application & parameter controls |
| **Visualization** | `Plotly`, `Matplotlib`, `Seaborn` | Interactive 2D/3D graphs, radar charts, membership contours |
| **Testing** | `pytest` | Unit testing and algorithm validation |

---

## 🚀 Execution & Setup Guide

### 1. Environment Setup
```bash
# Clone the repository and navigate into the project directory
cd "K and C mean"

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Running the Interactive Streamlit Application
```bash
streamlit run app.py
```

### 3. Running Unit Tests
```bash
pytest tests/
```

---

## 📊 Phase 1: Verified Dataset Inventory

All five benchmark datasets are downloaded, preprocessed, verified, and stored locally:

| # | Dataset | Raw Source & Filename | Raw Records & Dimensions | Clustering Records & Features | Excluded Identifiers & Target | Missing Values & Duplicates |
|---|---|---|---|---|---|---|
| **1** | **Credit Card Customers** | [Kaggle Dataset](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)<br>`BankChurners.csv` | **10,127** rows × **23** cols | **10,127** rows × **32** features | **ID:** `CLIENTNUM`<br>**Target:** `Attrition_Flag`<br>**Dropped:** 2 Naive Bayes artifact cols | **Missing:** 0<br>**Duplicates:** 0 |
| **2** | **Customer Personality Analysis** | [Kaggle Dataset](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)<br>`marketing_campaign.csv` | **2,240** rows × **29** cols | **2,240** rows × **34** features | **ID:** `ID`<br>**Target:** `Response`<br>**Dropped:** `Z_CostContact`, `Z_Revenue` | **Missing:** 24 (`Income`)<br>**Duplicates:** 0<br>*Dt_Customer converted to days* |
| **3** | **Wine Quality** | [Kaggle Dataset](https://www.kaggle.com/datasets/aleixdorca/wine-quality)<br>`winequality.csv` | **6,497** rows × **14** cols | **6,497** rows × **11** features | **Target:** `quality`<br>**Dropped:** `good`, `color` | **Missing:** 0<br>**Duplicates:** 1,177<br>*(Combined red & white verified)* |
| **4** | **Online Retail UCI** | [Kaggle Dataset](https://www.kaggle.com/datasets/jihyeseo/online-retail-data-set-from-uci-ml-repo)<br>`Online Retail.xlsx` | **541,909** transactions × **8** cols | **4,338** unique customer profiles × **3** features (`Recency`, `Frequency`, `Monetary`) | **ID:** `CustomerID`<br>**Target:** None (RFM Customer Segmentation) | **Missing:** 136,534 (`CustomerID`)<br>**Duplicates:** 5,268<br>*(Pruned cancellations & invalid prices)* |
| **5** | **Synthetic Classification 1000** | [Kaggle Competition](https://www.kaggle.com/competitions/syntheticdata1000)<br>`data_train_1000.csv` | **1,000** rows × **231** cols | **1,000** rows × **230** features (`F1` to `F230`) | **Target:** `Label`<br>**Excluded completely** from clustering features | **Missing:** 0<br>**Duplicates:** 0<br>*(Balanced 501 class 0 / 499 class 1)* |

---

## 📑 Next Phase

Phase 1 (Dataset Integration & Validation) is complete. Phase 2 will implement the core machine learning models (`src/models/`): K-Means and Fuzzy C-Means (FCM).

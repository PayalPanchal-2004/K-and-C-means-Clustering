"""EDA Visualisation Utilities for Clustering Datasets (Plotly & Matplotlib)."""

from typing import List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA


def plot_missing_values(df: pd.DataFrame) -> go.Figure:
    """Generate a Plotly bar chart displaying missing values per column."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        "Feature": missing.index,
        "Missing_Count": missing.values,
        "Percentage": missing_pct.values,
    }).sort_values(by="Missing_Count", ascending=False)

    fig = px.bar(
        missing_df,
        x="Feature",
        y="Missing_Count",
        hover_data=["Percentage"],
        title="Missing Values Analysis",
        color="Missing_Count",
        color_continuous_scale="Reds",
    )
    fig.update_layout(xaxis_tickangle=-45, template="plotly_white")
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    max_features: int = 20,
    title: str = "Feature Correlation Matrix",
) -> go.Figure:
    """Generate a Plotly heatmap for feature correlations."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] > max_features:
        # Select top variance features to keep heatmap readable
        variances = num_df.var().sort_values(ascending=False)
        selected_cols = variances.index[:max_features].tolist()
        num_df = num_df[selected_cols]

    corr = num_df.corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=f"{title} (Top {min(len(corr), max_features)} Features)",
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_outlier_boxplots(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    max_features: int = 8,
    title: str = "Feature Distributions & Outlier Inspection",
) -> go.Figure:
    """Generate interactive box plots to inspect data distribution and outliers."""
    num_df = df.select_dtypes(include=[np.number])
    if features is None:
        features = num_df.columns[:max_features].tolist()
    else:
        features = [f for f in features if f in num_df.columns][:max_features]

    fig = go.Figure()
    for feat in features:
        fig.add_trace(go.Box(y=num_df[feat], name=feat, boxpoints="outliers"))

    fig.update_layout(
        title=title,
        yaxis_title="Feature Value (Standardized / Raw)",
        template="plotly_white",
    )
    return fig


def plot_pca_2d(
    X: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "2D PCA Projection",
) -> go.Figure:
    """Project dataset into 2D using PCA and render an interactive Plotly scatter."""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_ * 100

    plot_df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "Cluster": [f"Cluster {l}" for l in labels] if labels is not None else "All Samples",
    })

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        title=f"{title} (Explained Variance: PC1={var_exp[0]:.1f}%, PC2={var_exp[1]:.1f}%)",
        template="plotly_white",
        opacity=0.7,
    )
    return fig


def plot_pca_3d(
    X: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "3D PCA Projection",
) -> go.Figure:
    """Project dataset into 3D using PCA and render an interactive 3D Plotly scatter."""
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_ * 100

    plot_df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "PC3": coords[:, 2],
        "Cluster": [f"Cluster {l}" for l in labels] if labels is not None else "All Samples",
    })

    fig = px.scatter_3d(
        plot_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Cluster",
        title=f"{title} (Total Variance: {sum(var_exp):.1f}%)",
        opacity=0.7,
    )
    return fig

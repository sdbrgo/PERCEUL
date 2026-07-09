import pandas as pd
import matplotlib.pyplot as plt
import joblib
from importlib import resources

from .utils.cluster import *
from .utils.auxiliary import build_umap, build_hdbscan, format_outliers, format_deviations_as_columns

from sklearn.cluster import KMeans

__all__ = [
    "explore_clusters",
    "final_clustering"
]

# ============================================================
#                       CLUSTER EXPLORATION 
# ============================================================
def explore_clusters(file: str):
    df = pd.read_csv(file)
    
    pkg_resource = resources.files("perceul.assets").joinpath("exploration_pipeline.pkl")
    
    with pkg_resource.open("rb") as f:
        exploration_pipeline = joblib.load(f)

    X_exp = exploration_pipeline.fit_transform(df)

    # dynamic UMAP constructor
    umap_model = build_umap(df) 
    X_umap = umap_model.fit_transform(X_exp)
    
    # dynamic HDBSCAN constructor
    hdb = build_hdbscan(df) 
    labels_hdb = hdb.fit_predict(X_umap)

    # --- cluster statistics ---
    from collections import Counter
    label_counts = Counter(labels_hdb)

    n_outliers = label_counts.pop(-1, 0)

    cluster_summary = {
        f"Cluster {key}": value
        for key, value in sorted(label_counts.items())
    }

    # --- visualization ---
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=labels_hdb,
        s=5,
        alpha=0.8
    )

    ax.set_title("Cluster Exploration of Worker Profiles")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    fig.tight_layout()
    
    return fig, cluster_summary, format_outliers(n_outliers)

# ============================================================
#                      FINAL CLUSTERING 
# ============================================================
def final_clustering(file: str, top_features: int) -> tuple:
    """
    Clusters data points using PCA + KMeans, identifies top drivers for each cluster.

    Args:
        file (str): Path to the input CSV file containing worker data.
        top_features (int): Number of top features to identify as drivers for each cluster.

    Returns:
        A tuple containing:
        * fig (matplotlib.figure.Figure): The PCA scatter plot figure with clusters colored.
        * best_k (int): optimal number of clusters determined by silhouette score.
        * deviations_markdown (str): markdown-formatted string showing top feature deviations for each cluster.
        * pca (sklearn.decomposition.PCA): The fitted PCA model.
        * feature_names (list): List of feature names corresponding to the original data.
    """
    df = pd.read_csv(file)
    
    pkg_resource = resources.files("perceul.assets").joinpath("core_pipeline.pkl")
    
    with pkg_resource.open("rb") as f:
        core_pipeline = joblib.load(f)

    X_pca = core_pipeline.fit_transform(df)

    best_k = choose_k(X_pca)                                                # choose_k() is from utils.cluster; dynamic `k` selection

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init="auto"
    )
    labels = kmeans.fit_predict(X_pca)

    # ===================================================
    #               Plotting PCA Clusters
    # ===================================================

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        s=5,
        alpha=0.8
    )
    ax.set_title("Final Clustering of Worker Profiles (PCA Space)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    fig.tight_layout()

    # ===================================================
    #               Cluster Analysis
    # ===================================================
    pca = core_pipeline.named_steps["pca"]                                  # Named Step Access to PCA model
    scaler = core_pipeline.named_steps["scaler"]                            # Named Step Access to Scaler model
    feature_names = df.columns.tolist()
    centroids = compute_cluster_centroids_pca(X_pca, labels)                # function is from `cluster_utils.py`; returns a DataFrame with PCA-space centroids for each cluster
    original_centroids = inverse_project_centroids(
        centroids,
        pca,
        scaler,
        feature_names
    )                                                                       # converts PCA-space centroids to original features; returns a DataFrame
    top_drivers = identify_top_drivers(original_centroids, top_features)    # returns a dict 
    deviations_markdown = format_deviations_as_columns(top_drivers)         # formats the output as a markdown table string

    return fig, best_k, deviations_markdown, pca, feature_names
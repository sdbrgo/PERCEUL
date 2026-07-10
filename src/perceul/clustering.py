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
def explore_clusters(file: str, include_summary: bool = False, include_markdown: bool = False):
    """
    Clusters data points using PCA + KMeans, identifies top drivers for each cluster.

    Args:
        file (str): Path to the input CSV file containing worker data.
        include_summary (bool): Whether to include cluster summary. (default: False)
        include_markdown (bool): Whether to include markdown-formatted outliers information. (default: False)

    Returns:
        results (dict): A dictionary containing the following keys:
        * plot (matplotlib.figure.Figure): The HDBSCAN scatter plot figure with clusters and outliers colored.
        * cluster_summary (dict): A dictionary containing cluster numbers (key) and number of data points within it (value).
        * outliers (str): Markdown-formatted string showing number of outliers.
    """

    df = pd.read_csv(file)
    results = {}
    

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

    # ---------- Storing values in the dictionary to return ----------

    results["plot"] = fig

    if include_summary:
        cluster_summary = {
            f"Cluster {key}": value
            for key, value in sorted(label_counts.items())
        }

        results["summary"] = cluster_summary

    if include_markdown:
        n_outliers = label_counts.pop(-1, 0)
        outliers_md = format_outliers(n_outliers)

        results["markdown"] = outliers_md
    
    return results

# ============================================================
#                      FINAL CLUSTERING 
# ============================================================
def final_clustering(file: str, top_features: int = 5, include_pca: bool = False, include_features: bool = False):
    """
    Clusters data points using PCA + KMeans, identifies top drivers for each cluster.

    Args:
        file (str): Path to the input CSV file containing worker data.
        top_features (int): Number of top features to identify as drivers for each cluster. (default:5)
        include_pca (bool): Whether to include the fitted PCA model in the results. (default: False)
        include_features (bool): Whether to include the original feature names in the results. (default: False)

    Returns:
        results (dict): A dictionary containing the following keys:
        * fig (matplotlib.figure.Figure): The PCA scatter plot figure with clusters colored.
        * best_k (int): optimal number of clusters determined by silhouette score.
        * deviations_md (str): markdown-formatted string showing top feature deviations for each cluster.
        * pca (sklearn.decomposition.PCA): The fitted PCA model.
        * feature_names (list): List of feature names corresponding to the original data.
    """
    df = pd.read_csv(file)

    results = {}
    
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

    results["plot"] = fig
    results["best_k"] = best_k
    results["deviations_md"] = deviations_markdown

    if include_pca:
        results["pca"] = pca

    if include_features:
        results["feature_names"] = feature_names

    return results
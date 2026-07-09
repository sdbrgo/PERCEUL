import numpy as np
import pandas as pd

import umap as UMAP
import hdbscan

def choose_umap_params(n_samples, n_features): # created to emphasize local structure of data
    # use simple heuristic
    sample_based = int(np.sqrt(n_samples) - 1)
    feature_based = int(np.log2(n_features) - 1)

    floor = 2
    min_n_neighbors = min(sample_based, feature_based)

    # find best_min_dist
    best_min_dist = 0.1 if n_features < 20 else 0.0 
    
    return max(floor, min_n_neighbors), best_min_dist

def build_umap(X):
    n_samples, n_features = X.shape

    # helper function defined above
    best_n_neighbors, best_min_dist = choose_umap_params(n_samples, n_features)
    
    return UMAP(
        n_neighbors=best_n_neighbors,
        min_dist=best_min_dist,
        n_components=2,
        random_state=42
    )

def build_hdbscan(X):
    n_samples = X.shape[0]

    min_cluster_size = int(max(2, 0.01 * n_samples))
    min_samples = max(2, int(0.5 * min_cluster_size))

    return hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom", # default
        prediction_data=True # default
    )

def format_outliers(n_outliers):
    if n_outliers == 0:
        return "✅ No outliers detected."
        
    return f"⚠️ **{n_outliers} workers** do not strongly belong to any cluster."
    

def format_deviations_as_columns(drivers: dict) -> str:
    """
    Format the top feature deviations for each cluster into a markdown table with clusters as columns.

    Args:
        drivers (dict): A dictionary where keys are cluster IDs and values are dictionaries containing feature deviations.

    Returns:
        str: A markdown table with clusters as columns and feature deviations as rows.
    """
    headers = []
    cells = []

    for cid, data in drivers.items():
        headers.append(f"Cluster {cid}")

        dev_lines = [
            f"{feature}: {value:.3f}"
            for feature, value in data["deviations"].items()
        ]

        cells.append("<br>".join(dev_lines))

    table = "| " + " | ".join(headers) + " |\n"
    table += "|" + "|".join(["---"] * len(headers)) + "|\n"
    table += "| " + " | ".join(cells) + " |"

    return table
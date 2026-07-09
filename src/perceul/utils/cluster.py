import pandas as pd 
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

__all__ = [
    "choose_k",
    "compute_cluster_centroids_pca",
    "inverse_project_centroids",
    "compute_cluster_stats",
    "identify_top_drivers"
]

#========== Before Final Clustering ========== 
def choose_k(X_pca):
    best_k = 2
    best_score = -1
    
    # Ensure k does not exceed n_samples - 1 for silhouette_score validity
    n_samples = X_pca.shape[0]
    max_k_for_silhouette = n_samples # range is exclusive of end, so this will allow k up to n_samples - 1

    for k in range(2, min(12, max_k_for_silhouette)):
        km = KMeans(n_clusters=k, random_state=42, n_init='auto') # Added n_init='auto' to suppress future warning
        labels = km.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)

        if score > best_score:
            best_score = score
            best_k = k

    print(f"Executing choose_k()... Best Score: {best_score}")

    return best_k

#========== During Cluster Analysis ========== 
def compute_cluster_centroids_pca(df_pca: pd.DataFrame, labels: list) -> pd.DataFrame:
    """
    Compute cluster centroids in PCA space.

    Args:
        df_pca (pd.DataFrame): DataFrame with PCA-transformed data.
        labels (list): List of cluster labels for each data point.

    Returns:
        pd.DataFrame: DataFrame with cluster centroids.
    """
    df = pd.DataFrame(df_pca)
    df['cluster'] = labels

    return df.groupby('cluster').mean()

# maps PCA-space centroids back to original feature space
def inverse_project_centroids(pca_centroids, pca_model, scaler_model, original_feature_names) -> pd.DataFrame:
    """
    Inverse project PCA-space centroids back to original feature space.

    Args:
        pca_centroids (pd.DataFrame): DataFrame with PCA-space centroids.
        pca_model (sklearn.decomposition.PCA): Fitted PCA model.
        scaler_model (sklearn.preprocessing.StandardScaler): Fitted scaler model.
        original_feature_names (list): List of original feature names.

    Returns:
        pd.DataFrame: DataFrame with centroids in original feature space.
    """

    scaled_centroids = pca_model.inverse_transform(pca_centroids.values) # back-project from PCA space to scaled feature space
    original_space_centroids = scaler_model.inverse_transform(scaled_centroids) # undo scaling

    df_original = pd.DataFrame(
        original_space_centroids,
        columns=original_feature_names,
        index=pca_centroids.index
    )

    return df_original

# function to compute and save cluster stats
def compute_cluster_stats(df_pca, labels, feature_names):
    df = pd.DataFrame(df_pca, columns=feature_names)
    df['cluster'] = labels

    stats = {}

    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id].drop(columns=['cluster'])

        stats[cluster_id] = {
            "count": len(cluster_data),
            "mean": cluster_data.mean().to_dict(),
            "median": cluster_data.median().to_dict(),
            "std": cluster_data.std().to_dict(),
            "min": cluster_data.min().to_dict(),
            "max": cluster_data.max().to_dict(),
            "range": (cluster_data.max() - cluster_data.min()).to_dict()
        }

    return stats

# ranks top drivers based on `top_n`
def identify_top_drivers(original_space_centroids: pd.DataFrame, top_n: int) -> dict:
    global_mean = original_space_centroids.mean()

    drivers = {}

    for cluster_id, row in original_space_centroids.iterrows():
        deviation = row - global_mean
        ranked = deviation.abs().sort_values(ascending=False)

        top_features = ranked.head(top_n).index.tolist()

        drivers[cluster_id] = {
            # "top_features": top_features,
            "deviations": deviation[top_features].to_dict()
        }

    return drivers
import pandas as pd
from sklearn.decomposition import PCA 

# ============================================================
# A function to retreive the loadings of the PCA components
# ============================================================

def get_pca_loadings(pca: PCA, feature_names: list) -> pd.DataFrame:
    """
    Get loadings of PCA components.

    Args:
        pca (sklearn.decomposition.PCA): PCA object from sklearn
        feature_names (list): List of feature names corresponding to the original data

    Returns:
        loadings (pd.DataFrame): DataFrame with PCA loadings for each component and feature
    """
    loadings = pd.DataFrame(
        pca.components_.T,                                          # transposes the get features as rows and components as columns
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],     # names the columns as PC1, PC2, ...
        index=feature_names                                         # names the rows with the original feature names
    )

    # Convert the index into a visible column named 'Feature'
    loadings = loadings.reset_index().rename(columns={'index': 'Feature'})

    return loadings

# For version 0.6.0, add a function to manually map indexes/row of the original dataset to the clusters of either the exploration or final clustering.
# This is useful to know which workers belong to which cluster, especially for the final clustering where the number of clusters is determined dynamically.
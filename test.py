import matplotlib.pyplot as plt
from perceul import explore_clusters

# 1. Load your raw dataset (can contain mixed data types)
data_path = r"C:\Users\JAZ\OneDrive\Documents\GitHub\perceul\perceul-data - TACTICS.csv"

# 2. Execute the pipeline
# This automatically strips non-numeric features, handles missing values, 
# scales your data, projects it via PCA, and performs clustering after selecting the optial number of clusters.
# It returns a figure (matplotlib), optimal number of clusters (k), deviations in markdown, 
# the PCA model, and the top features characterizing each cluster.
output = explore_clusters(data_path, include_summary=True, include_markdown=True)

# 3. Display the plotted figure
plt.figure(output["plot"].number)
plt.show()

print(output["summary"])

print(output["markdown"])
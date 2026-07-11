# PERCEUL
A Framework and Prototype for Unsupervised Profiling of Perception and Cognitive Ergonomics in the Workplace

![PERCEUL]()

---

## 📌 Overview

`perceul` aims to bridge ergonomics and AI, with the project using Unsupervised Learning for clustering latent perceptual profiles of workers in different work settings. It can be used by Ergonomics Analysts and Consultants who are sufficiently knowledgeable on Unsupervised Learning implementation, workflow, and cluster interpretation. The project includes using psychological, perceptual, self-reporting scales or tools, and Unsupervised Learning algorithms and workflow. PERCEUL exists to address the gap in AI and ergonomics, serving as an innovation that automates the profiling of workers for workspace reorganization, resource allocation, and strategic workplace dynamics, ultimately becoming a decision-support tool.

---

## ⚒️ Quickstart: Running the Clustering Pipeline

`perceul` abstracts away complex data preprocessing by embedding a native cleaning and dimensionality reduction pipeline. Here is how to load data, process features, and determine clusters in just a few lines of code:

```python
import matplotlib.pyplot as plt
from perceul import final_clustering

# 1. Load your raw dataset (can contain mixed data types)
data_path = "your_data.csv"

# 2. Execute the pipeline
# This automatically strips non-numeric features, handles missing values, 
# scales your data, projects it via PCA, and performs clustering after selecting the optial number of clusters.
# It returns a figure (matplotlib), optimal number of clusters (k), deviations in markdown, 
# the PCA model, and all features within the dataset.
results = final_clustering(data_path, top_features=5)

# 3. Display the plotted figure
plt.figure(results["plot"].number)
plt.show()
```

--- 

## 📄 License

This project is licensed under the **MIT License**, which allows commercial and non-commercial use, modification, and distribution, provided that proper attribution is given.

See the full license text in the [`LICENSE`](https://github.com/jasper-gomez/perceul/blob/refactor/package/LICENSE) file.
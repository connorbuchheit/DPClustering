from dbscan import dbscan
from k_means import k_means
from generate_data import generate_clusters
from add_noise_to_data import synth_data
import matplotlib.pyplot as plt

# Generate clusters + run algorithms
X = generate_clusters(n_clusters=3, points_per_cluster=100, dim=2, cluster_std=0.5, separation=5)
dbscan_labels = dbscan(X, eps=0.50, min_samples=5)
_, kmeans_labels = k_means(X, k=3)

# Step 4: Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='tab10', s=15)
plt.title("DBSCAN Clustering")
plt.xlabel("X0")
plt.ylabel("X1")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='tab10', s=15)
plt.title("k-Means Clustering")
plt.xlabel("X0")
plt.ylabel("X1")

plt.tight_layout()
plt.show()
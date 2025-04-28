from dbscan import dbscan
from k_means import k_means
from dp_k_means import add_noise_to_centroids_and_relabel
from generate_data import generate_clusters
#from add_noise_to_data import synth_data  --> (pretty sure isn't used in this flow currently, uncomment if needed)
import matplotlib.pyplot as plt
import numpy as np #needed for non-DP DBSCAN

# parameters for clustering
N_CLUSTERS = 3
POINTS_PER_CLUSTER = 100
DIM = 2
CLUSTER_STD = 0.5
SEPARATION = 5
EPSILON = 1.0
DATA_BOUND_B = 10.0

# Generate clusters + run algorithms
# X = generate_clusters(n_clusters=3, points_per_cluster=100, dim=2, cluster_std=0.5, separation=5)
# dbscan_labels = dbscan(X, eps=0.50, min_samples=5)
# _, kmeans_labels = k_means(X, k=3)

X = generate_clusters(
    n_clusters=N_CLUSTERS,
    points_per_cluster=POINTS_PER_CLUSTER,
    dim=DIM,
    cluster_std=CLUSTER_STD,
    separation=SEPARATION
)

dbscan_labels = dbscan(X, eps=0.50, min_samples=5) #Non-DP Algorithms
original_centroids, kmeans_labels = k_means(X, k=N_CLUSTERS) #Non-DP k-Means

dp_centroids, dp_kmeans_labels = add_noise_to_centroids_and_relabel(
    X, original_centroids, epsilon=EPSILON, b=DATA_BOUND_B
)


# Plot results
plt.figure(figsize=(18, 5)) #12,5 --> 18,5 to accommodate third plot

# a. non-DP DBSCAN
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='tab10', s=15, alpha=0.7)
plt.title("Non-DP DBSCAN Clustering")
plt.xlabel("X0")
plt.ylabel("X1")
noise_points = X[np.array(dbscan_labels) == -1]
plt.scatter(noise_points[:, 0], noise_points[:, 1], color='black', s=5, marker='x', label='Noise')
plt.legend()

# b. non-DP k-Means
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='tab10', s=15, alpha=0.7)
plt.scatter(original_centroids[:, 0], original_centroids[:, 1], marker='X', s=100, c='red', label='Original Centroids')
plt.title("Non-DP k-Means Clustering")
plt.xlabel("X0")
plt.ylabel("X1")
plt.legend()

# c. DP k-Means with noisy centroids
plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=dp_kmeans_labels, cmap='tab10', s=15, alpha=0.7)
plt.scatter(dp_centroids[:, 0], dp_centroids[:, 1], marker='P', s=100, c='blue', label='DP Centroids') # 'P' for plus (filled)
plt.scatter(original_centroids[:, 0], original_centroids[:, 1], marker='X', s=100, c='red', alpha=0.5, label='Original Centroids') # Show original for comparison
plt.title(f"DP k-Means (Noisy Centroids, $\epsilon$={EPSILON}, b={DATA_BOUND_B})")
plt.xlabel("X0")
plt.ylabel("X1")
plt.legend()

plt.tight_layout()
plt.show()
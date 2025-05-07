from dpclustering.data import generate_clusters
from dpclustering.kmeans import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Parameters
N_CLUSTERS = 3
POINTS_PER_CLUSTER = 500
DIM = 2
CLUSTER_STD = 1.0
SEPARATION = 5
EPSILON = 10
DATA_BOUND_B = 8.0

# Generate data
X, _ = generate_clusters(
    n_clusters=N_CLUSTERS,
    points_per_cluster=POINTS_PER_CLUSTER,
    dim=DIM,
    cluster_std=CLUSTER_STD,
    separation=SEPARATION
)
X = np.clip(X, -DATA_BOUND_B, DATA_BOUND_B)

# a. Original K-Means
kmeans = KMeans(X, N_CLUSTERS, DATA_BOUND_B)
original_centroids = kmeans.fit()[0]
original_labels = np.argmin(np.linalg.norm(X[:, None, :] - original_centroids[None, :, :], axis=2), axis=1)


# b. DP KMeans using Kaplan-Stemmer
dp_kmeans = KMeans(X, N_CLUSTERS, DATA_BOUND_B)
dp_centroids = dp_kmeans.kaplan_stemmer_fit(EPSILON)

# Predict labels for plotting
dp_labels = np.argmin(np.linalg.norm(X[:, None, :] - dp_centroids[None, :, :], axis=2), axis=1)

# Plotting
cmap = 'tab10'
point_size = 10
alpha_val = 0.5
plt.figure(figsize=(16, 6))
plot_lim = (-DATA_BOUND_B - 1, DATA_BOUND_B + 1)

# a. Original
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=original_labels, cmap=cmap, s=point_size, alpha=alpha_val)
plt.scatter(original_centroids[:, 0], original_centroids[:, 1], marker='X', s=120, c='black', label='Original')
plt.title("Original K-Means")
plt.xlim(*plot_lim)
plt.ylim(*plot_lim)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()

# b. DP
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=dp_labels, cmap=cmap, s=point_size, alpha=alpha_val)
plt.scatter(dp_centroids[:, 0], dp_centroids[:, 1], marker='P', s=150, c='black', edgecolors='white', linewidth=1, label='DP Centers')
plt.title(rf"Kaplan-Stemmer DP K-Means ($\epsilon$={EPSILON})")
plt.xlim(*plot_lim)
plt.ylim(*plot_lim)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()

plt.tight_layout()
plt.show()

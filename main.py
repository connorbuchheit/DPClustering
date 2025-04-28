from k_means import k_means
from dp_k_means import add_noise_to_centroids_and_relabel, add_noise_to_input_data
from generate_data import generate_clusters
import matplotlib.pyplot as plt
import numpy as np

# parameters for the clustering
N_CLUSTERS = 3
POINTS_PER_CLUSTER = 500
DIM = 2
CLUSTER_STD = 1.0 
SEPARATION = 5
EPSILON = 0.1
DATA_BOUND_B = 8.0

X = generate_clusters(
    n_clusters=N_CLUSTERS,
    points_per_cluster=POINTS_PER_CLUSTER,
    dim=DIM,
    cluster_std=CLUSTER_STD,
    separation=SEPARATION
)
X = np.clip(X, -DATA_BOUND_B, DATA_BOUND_B)

# algoritms
# a. original K-Means (non-DP)
original_centroids, original_labels = k_means(X, k=N_CLUSTERS)

# b. DP K-Means --> add noise to centroids (output perturbation on centroids)
dp_centroids_noisy, dp_labels_noisy_centroids = add_noise_to_centroids_and_relabel(
    X, original_centroids, original_labels, epsilon=EPSILON, b=DATA_BOUND_B
)

# c. DP --> process via input perturbation (add noise to data, then cluster)
noisy_X_input_perturb = add_noise_to_input_data(
    X, epsilon=EPSILON, b=DATA_BOUND_B
)
centroids_from_noisy_data, labels_from_noisy_data = k_means(
    noisy_X_input_perturb, k=N_CLUSTERS
)

# plotting
cmap = 'tab10'
point_size = 10
alpha_val = 0.5
plt.figure(figsize=(24, 6))
plot_lim_x_bounded = (-DATA_BOUND_B - 1, DATA_BOUND_B + 1)
plot_lim_y_bounded = (-DATA_BOUND_B - 1, DATA_BOUND_B + 1)

# a. original K-Means
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=original_labels, cmap=cmap, s=point_size, alpha=alpha_val) # Use updated size/alpha
plt.scatter(original_centroids[:, 0], original_centroids[:, 1], marker='X', s=120, c='black', label='Original Centroids')
plt.title("Original K-Means")
plt.xlabel("X0")
plt.ylabel("X1")
plt.xlim(plot_lim_x_bounded)
plt.ylim(plot_lim_y_bounded)
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')

# b. add noise to centroids
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=dp_labels_noisy_centroids, cmap=cmap, s=point_size, alpha=alpha_val) # Use updated size/alpha
plt.scatter(dp_centroids_noisy[:, 0], dp_centroids_noisy[:, 1],
            marker='P', s=150, c='black', edgecolors='white', linewidth=1,
            label=f'DP Centroids ($\epsilon$={EPSILON})')
plt.title("Add Noise to Centroids")
plt.xlabel("X0")
plt.ylabel("X1")
plt.xlim(plot_lim_x_bounded)
plt.ylim(plot_lim_y_bounded)
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')

# c. original data labeled by clustering noisy data
plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels_from_noisy_data, cmap=cmap, s=point_size, alpha=alpha_val)
plt.title(f"Original Data Labeled via Noisy Data ($\epsilon$={EPSILON})")
plt.xlabel("X0")
plt.ylabel("X1")
plt.xlim(plot_lim_x_bounded)
plt.ylim(plot_lim_y_bounded)
plt.gca().set_aspect('equal', adjustable='box')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
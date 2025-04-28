import numpy as np

# funct. for adding noise to centroids (output perturbation on centroids)
def add_noise_to_centroids_and_relabel(X, centroids, labels, epsilon, b):
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive.")
    if b <= 0:
        raise ValueError("Data bound 'b' must be positive.")

    k, n_features = centroids.shape
    if k == 0:
        return np.copy(centroids), np.copy(labels)

    noisy_centroids = np.zeros_like(centroids)

    if k > 0:
        epsilon_per_centroid = epsilon / k
    else:
        epsilon_per_centroid = epsilon

    for i in range(k):
        cluster_points = X[labels == i]
        cluster_size = len(cluster_points) #gets size

        if cluster_size > 0:
            sensitivity = (2.0 * b) / cluster_size

            if epsilon_per_centroid > 0:
                 scale = sensitivity / epsilon_per_centroid
            else:
                 scale = np.inf

            noise = np.random.laplace(0, scale, size=centroids[i].shape)
            noisy_centroids[i] = centroids[i] + noise
        else:
            noisy_centroids[i] = centroids[i]

    distances = np.linalg.norm(X[:, np.newaxis] - noisy_centroids, axis=2)
    noisy_labels = np.argmin(distances, axis=1)

    return noisy_centroids, noisy_labels

# funct. for adding noise to data (input perturbation)
def add_noise_to_input_data(X, epsilon, b):

    if epsilon <= 0:
        raise ValueError("Epsilon must be positive.")
    if b <= 0:
        raise ValueError("Data bound 'b' must be positive.")

    # sensitivity for each coordinate assuming data bounded in [-b, b]
    sensitivity_per_coordinate = 2.0 * b
    scale = sensitivity_per_coordinate / epsilon

    noise = np.random.laplace(0, scale, size=X.shape)
    noisy_X = X + noise

    return noisy_X
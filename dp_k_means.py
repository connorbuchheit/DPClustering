import numpy as np

def add_noise_to_centroids_and_relabel(X, centroids, epsilon, b):
    # applies DP to k-means centroids via the Laplace mechanism
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive.")
    if b <= 0:
        raise ValueError("Data bound 'b' must be positive.")

    k, n_features = centroids.shape
    sensitivity = 2.0 * b

    # after the standard k-means algorithm has run. then re-assigns labels based on the noisy centroids.
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, size=centroids.shape)
    noisy_centroids = centroids + noise
    distances = np.linalg.norm(X[:, np.newaxis] - noisy_centroids, axis=2)

    noisy_labels = np.argmin(distances, axis=1)
    return noisy_centroids, noisy_labels
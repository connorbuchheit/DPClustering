# NON-DP k-means
import numpy as np

def k_means(X, k, max_iter=100, tol=1e-4):
    """
    Perform k-means clustering on the dataset X.

    Parameters:
    - X: numpy array of shape (n_samples, n_features)
    - k: number of clusters
    - max_iter: maximum number of iterations

    Returns:
    - centroids: final cluster centroids
    - labels: cluster labels for each sample
    """

    # Randomly initialize centroids
    n_samples, n_features = X.shape
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices]

    for _ in range(max_iter):
        # Compute distances from samples to centroids
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

        # Assign each sample to the nearest centroid
        labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(np.abs(new_centroids - centroids) < tol):
            break

        centroids = new_centroids

    return centroids, labels
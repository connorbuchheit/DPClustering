import numpy as np

class KMeans:
    def __init__(self, X, k, b, max_iter=100, tol=1e-4):
        """
        Initialize the KMeans clustering algorithm.

        Parameters:
        X (numpy.ndarray): The input X for clustering.
        k (int): The number of clusters.
        b (float): The bounds for clipping the data points ([-b, b]^N).
        max_iter (int): The maximum number of iterations.
        tol (float): The tolerance for convergence.
        """
        self.X = X
        self.k = k
        self.b = b
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self):
        n_samples, _ = self.X.shape
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        centroids = self.X[random_indices]

        for _ in range(self.max_iter):
            # Compute distances from samples to centroids
            distances = np.linalg.norm(self.X[:, np.newaxis] - centroids, axis=2)

            # Assign each sample to the nearest centroid
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([self.X[labels == i].mean(axis=0) for i in range(self.k)])

            # Check for convergence
            if np.all(np.abs(new_centroids - centroids) < self.tol):
                break

            centroids = new_centroids

        self.centroids = centroids
        self.labels = labels
        return self.centroids, self.labels
    
    def add_noise_to_centroids(self, epsilon):
        """
        Finds the centroids of each cluster, adds Laplace noise to them, then
        reassigns the labels of the data points to the noisy centroids. Must
        have called fit() first, which computes the centroids.

        Parameters:
        epsilon (float): The privacy budget for differential privacy.
        
        Returns:
        noisy_centroids (numpy.ndarray): The noisy centroids after adding noise.
        labels (numpy.ndarray): The labels of the data points after reassigning to noisy centroids.
        """
        if self.centroids is None or self.labels is None:
            raise ValueError("Centroids have not been computed. Call fit() first.")

        noisy_centroids = np.zeros_like(self.centroids)
        for i in range(self.k):
            cluster_size = np.sum(self.labels == i)
            if cluster_size > 0:
                epsilon_per_centroid = epsilon / self.k
                sensitivity = 2 * self.b / cluster_size
                noise = np.random.laplace(0, sensitivity / epsilon_per_centroid, self.centroids[i].shape)
                noisy_centroids[i] = self.centroids[i] + noise
            else:
                noisy_centroids[i] = self.centroids[i]

        # reassign labels based on noisy centroids
        distances = np.linalg.norm(self.X[:, np.newaxis] - noisy_centroids, axis=2)
        self.labels = np.argmin(distances, axis=1)

        return noisy_centroids, self.labels
        
        
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

        self.n_points = X.shape[0]

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
    
    def add_noise_to_centroids(self, epsilon, release_labels=True):
        """
        Finds the centroids of each cluster, adds Laplace noise to them, then
        reassigns the labels of the data points to the noisy centroids, if 
        requested. Must have called fit() first, which computes the centroids.

        Parameters:
        epsilon (float): The privacy budget for differential privacy.
        
        Returns:
        noisy_centroids (numpy.ndarray): The noisy centroids after adding noise.
        labels (numpy.ndarray): The labels of the data points after reassigning to noisy centroids.
        """
        if self.centroids is None or (self.labels is None and release_labels):
            raise ValueError("Centroids have not been computed. Call fit() first.")

        if release_labels:
            centroid_epsilon = epsilon / 10
            label_epsilon = epsilon - centroid_epsilon
        else:
            centroid_epsilon = epsilon
        
        cluster_sizes = np.array([np.sum(self.labels == i) for i in range(self.k)])
        sensitivity = 2 * self.b / cluster_sizes
        scales = sensitivity / centroid_epsilon

        noise = np.random.laplace(0, scales[:, np.newaxis], size=self.centroids.shape)
        noisy_centroids = self.centroids + noise

        # reassign labels using exponential function
        if release_labels:
            sensitivity = 2 * self.b 

            labels = np.zeros(self.n_points, dtype=int)
            for i in range(self.n_points):
                utilities = -np.linalg.norm(noisy_centroids - self.X[i], axis=1)
                scores = np.exp((label_epsilon * utilities) / sensitivity)
                probs = scores / np.sum(scores)
                labels[i] = np.random.choice(self.k, p=probs)
            
            self.labels = labels

        # reassign labels using private distance
        # distances = np.linalg.norm(self.X[:, np.newaxis] - noisy_centroids, axis=2)
        # labels = np.argmin(distances, axis=1)

        
        return noisy_centroids, labels if release_labels else noisy_centroids
        
    def predict(self, x):
        """
        Predict the cluster for a new data point.

        Parameters:
        x (numpy.ndarray): The new data point to predict.

        Returns:
        int: The predicted cluster label.
        """
        if self.centroids is None:
            raise ValueError("Centroids have not been computed. Call fit() first.")

        distances = np.linalg.norm(self.centroids - x, axis=1)
        return np.argmin(distances)
        
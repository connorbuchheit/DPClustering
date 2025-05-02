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

        self.n_points, self.dims = X.shape
        

    def fit(self, init_centroids=None):
        if init_centroids is None:
            random_indices = np.random.choice(self.n_points, self.k, replace=False)
            centroids = self.X[random_indices]
        else:
            centroids = init_centroids

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
            labels = self._assign_labels(noisy_centroids, label_epsilon)
        
        return noisy_centroids, labels if release_labels else noisy_centroids
    
    def private_kmeans(self, epsilon, delta, phi2):
        """
        An implementation of the private KMeans algorithm. Additionally, it
        reassigns the labels of the data points to the noisy centroids using a
        helper function.
        Source: Algorithm 1 in https://dl.acm.org/doi/pdf/10.1145/3196959.3196977

        Parameters:
        epsilon (float): The privacy budget for differential privacy.
        delta (float): The probability of failure.
        phi2 (float): Separation parameter. Said to be <= 1/1000.

        Returns:
        noisy_centroids (numpy.ndarray): The noisy centroids after adding noise.
        labels (numpy.ndarray): The labels of the data points after reassigning to noisy centroids.
        """
        from sklearn.cluster import KMeans

        epsilon_centroids = epsilon / 10
        epsilon_assignment = epsilon - epsilon_centroids

        epsilon_prime = epsilon / (8 * self.k * np.sqrt(self.k * np.log(1 / delta)))
        delta_prime = delta / (8 * self.k**2)

        t = int(np.ceil(self.k * np.power(self.n_points, 1/10) * \
                        np.power(self.dims, 1/2) / epsilon_prime))
        t = min(t, self.n_points // self.k)
        subsets = np.array_split(self.X, t)

        subset_centroids = []
        for subset in subsets:
            km = KMeans(n_clusters=self.k, max_iter=1000, n_init=1)
            km.fit(subset)
            subset_centroids.append(km.cluster_centers_)
        subset_centroids = np.array(subset_centroids)

        aligned_centers = np.mean(subset_centroids, axis=0)

        sensitivity = phi2
        noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta_prime)) / epsilon_centroids
        noisy_centroids = aligned_centers + np.random.laplace(0, noise_scale, size=aligned_centers.shape)

        return noisy_centroids, self._assign_labels(noisy_centroids, epsilon_assignment)
        
    def _assign_labels(self, centroids, epsilon):
        """
        Assigns labels using the exponential mechanism. Assumes data ∈ [-b, b]^N.

        Parameters:
        centroids (numpy.ndarray): The centroids of the clusters (noisy OK).
        epsilon (float): The privacy budget for differential privacy.

        Returns:
        labels (numpy.ndarray): The labels of the data points after reassigning to noisy centroids.
        """
        distances = np.linalg.norm(self.X[:, np.newaxis] - centroids, axis=2)
        sensitivity = 2 * self.b * np.sqrt(self.dims)  # 2b√d
        scale = epsilon / (2 * sensitivity)

        scores = -distances
        max_scores = np.max(scale * scores, axis=1, keepdims=True)
        probabilities = np.exp(scale * scores - max_scores) # numerical stability
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        labels = np.array([np.random.choice(self.k, p=prob) for prob in probabilities])
        return labels

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
        
    
    
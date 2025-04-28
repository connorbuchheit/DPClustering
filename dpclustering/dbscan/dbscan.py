import numpy as np

class DBSCAN:
    def __init__(self, X, b, radius, min_samples, max_iter=100):
        """
        Initialize the DBSCAN clustering algorithm.

        Parameters:
        X (numpy.ndarray): The input X for clustering.
        b (float): The bounds for clipping the data points ([-b, b]^N).
        radius (float): The radius for neighborhood search.
        min_samples (int): The minimum number of samples in a neighborhood to form a core point.
        max_iter (int): The maximum number of iterations.
        """
        self.X = X
        self.b = b
        self.radius = radius
        self.min_samples = min_samples
        self.max_iter = max_iter

        self.n_points = X.shape[0]
        
    def fit(self, distance_func=None, density_func=None):
        """
        Perform DBSCAN clustering.

        Parameters:
            distance_func (callable, optional): Function to compute distance between two points.
            density_func (callable, optional): Function to compute density based on neighbors.

        Returns:
            list: Cluster labels for each point (-1 for noise).
        """
        if distance_func is None:
            distance_func = self._euclidean_distance
        if density_func is None:
            density_func = self._default_density

        def region_query(point_idx):
            """Find all neighbors within radius distance of a point."""
            neighbors = []
            for idx, point in enumerate(self.X):
                if distance_func(self.X[point_idx], point) <= self.radius:
                    neighbors.append(idx)
            return neighbors

        def expand_cluster(point_idx, neighbors, cluster_id):
            """Expand the cluster recursively."""
            clusters[point_idx] = cluster_id
            i = 0
            while i < len(neighbors):
                neighbor_idx = neighbors[i]
                if clusters[neighbor_idx] == -1:  # Previously marked as noise
                    clusters[neighbor_idx] = cluster_id
                elif clusters[neighbor_idx] == 0:  # Not yet visited
                    clusters[neighbor_idx] = cluster_id
                    new_neighbors = region_query(neighbor_idx)
                    if density_func(new_neighbors) >= self.min_samples / (np.pi * self.radius**2):
                        neighbors += new_neighbors
                i += 1

        # Initialize cluster labels (0 = unvisited, -1 = noise)
        clusters = [0] * len(self.X)
        cluster_id = 0

        # Iterate through each point
        for point_idx in range(len(self.X)):
            if clusters[point_idx] != 0:
                continue
            neighbors = region_query(point_idx)
            if density_func(neighbors) < self.min_samples / (np.pi * self.radius**2):
                clusters[point_idx] = -1  # Mark as noise
            else:
                cluster_id += 1
                expand_cluster(point_idx, neighbors, cluster_id)

        self.labels_ = clusters
        return clusters  

    def add_noise_to_densities(self, epsilon):
        """
        Adds Laplace noise to the density of each point in the dataset.

        Parameters:
        epsilon (float): The privacy budget for differential privacy.

        Returns:
        labels (numpy.ndarray): The labels of the data points after adding noise to densities.
        """
        epsilon_per_point = epsilon / self.n_points
        return self.fit(density_func=self._dp_density(epsilon=epsilon_per_point))
    
    def add_noise_to_distances(self, epsilon):
        """
        Adds Laplace noise to the distances between points in the dataset.

        Parameters:
        epsilon (float): The privacy budget for differential privacy.

        Returns:
        labels (numpy.ndarray): The labels of the data points after adding noise to distances.
        """
        epsilon_per_point = epsilon / self.n_points
        return self.fit(distance_func=self._dp_distance(epsilon=epsilon_per_point))

    def _euclidean_distance(self, point1, point2):
        """
        Compute the Euclidean distance between two points.

        Parameters:
        point1 (numpy.ndarray): The first point.
        point2 (numpy.ndarray): The second point.

        Returns:
        float: The Euclidean distance between the two points.
        """
        return np.linalg.norm(point1 - point2)

    def _default_density(self, neighbors):
        """
        Compute the density of a point based on its neighbors.

        Parameters:
        point (numpy.ndarray): The point for which to compute density.
        neighbors (list): The list of neighboring points.

        Returns:
        float: The density of the point.
        """
        return len(neighbors) / (np.pi * (self.radius ** 2))
    
    def _dp_distance(self, epsilon):
        """
        Compute the distance between two points with differential privacy.

        Parameters:
        point1 (numpy.ndarray): The first point.
        point2 (numpy.ndarray): The second point.
        epsilon (float): The privacy budget for differential privacy.

        Returns:
        float: The distance between the two points with added noise.
        """
        def dp_distance_aux(point1, point2):
            distance = self._euclidean_distance(point1, point2)
            sensitivity = 2 * self.b  # diameter of the bounding box [-b, b]
            noise = np.random.laplace(0, sensitivity/epsilon)
            return distance + noise
        
        return dp_distance_aux
    
    def _dp_density(self, epsilon):
        """
        Compute the density of a point based on its neighbors with differential privacy.

        Parameters:
        neighbors (numpy.ndarray): The list of neighboring points.
        epsilon (float): The privacy budget for differential privacy.

        Returns:
        float: The density of the point with added noise.
        """
        def dp_density_aux(neighbors):
            density = len(neighbors) / (np.pi * (self.radius ** 2))
            sensitivity = 1 / (np.pi * (self.radius ** 2))
            noise = np.random.laplace(0, sensitivity/epsilon)
            return density + noise
        return dp_density_aux
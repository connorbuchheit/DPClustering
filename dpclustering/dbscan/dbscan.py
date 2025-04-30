import numpy as np

class DBSCAN:
    def __init__(self, X, b, radius, min_samples, max_iter=2000):
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
        
    def fit(self, neighbor_matrix=None):
        """
        Perform DBSCAN clustering.

        Parameters:
        neighbor_matrix (numpy.ndarray): Optional precomputed privatized neighbor matrix. Defaults to Euclidean distance.

        Returns:
            list: Cluster labels for each point (-1 for noise).
        """
        if neighbor_matrix is None:
            # Standard DBSCAN with Euclidean distance
            def region_query(point_idx):
                neighbors = []
                for idx in range(self.n_points):
                    if self._euclidean_distance(point_idx, idx) <= self.radius:
                        neighbors.append(idx)
                return neighbors
        else:
            def region_query(point_idx):
                neighbors = []
                for idx in range(self.n_points):
                    if neighbor_matrix[point_idx, idx] == 1:
                        neighbors.append(idx)
                return neighbors

        def expand_cluster(point_idx, neighbors, cluster_id):
            """Expand the cluster recursively."""
            clusters[point_idx] = cluster_id
            i = 0
            while i < len(neighbors):
                neighbor_idx = neighbors[i]
                if clusters[neighbor_idx] == -1:
                    clusters[neighbor_idx] = cluster_id
                elif clusters[neighbor_idx] == 0:
                    clusters[neighbor_idx] = cluster_id
                    new_neighbors = region_query(neighbor_idx)
                    if len(new_neighbors) >= self.min_samples:
                        neighbors += [n for n in new_neighbors if n not in neighbors]
                i += 1

        # Initialize cluster labels (0 = unvisited, -1 = noise)
        clusters = [0] * self.n_points
        cluster_id = 0

        # Iterate through each point
        for point_idx in range(self.n_points):
            if clusters[point_idx] != 0:
                continue
            neighbors = region_query(point_idx)
            if len(neighbors) < self.min_samples:
                clusters[point_idx] = -1
            else:
                cluster_id += 1
                expand_cluster(point_idx, neighbors, cluster_id)

        self.labels_ = clusters
        return clusters 

    # def add_noise_to_densities(self, epsilon):
    #     """
    #     Adds Laplace noise to the density of each point in the dataset. Also 
    #     adds noise to the distances between points, because the density is 
    #     computed based on distances, which means a non-DP protected distance
    #     function would leak information about the data. Splits the budget
    #     equally between the two functions.

    #     Parameters:
    #     epsilon (float): The privacy budget for differential privacy.

    #     Returns:
    #     labels (numpy.ndarray): The labels of the data points after adding noise to densities.
    #     """
    #     half_epsilon = epsilon / 2
    #     sensitivity = 1

    #     dp_distance_func, distance_matrix = self._generate_noisy_distances(half_epsilon)

    #     densities = np.zeros(self.n_points)
    #     for i in range(self.n_points):
    #         for j in range(self.n_points):
    #             if distance_matrix[i, j] <= self.radius:
    #                 densities[i] += 1

    #     noise = np.random.laplace(0, sensitivity / half_epsilon, self.n_points)
    #     noisy_densities = densities + noise

    #     def dp_density(i):
    #         return noisy_densities[i]

    #     return self.fit(distance_func=dp_distance_func, density_func=dp_density)
    
    # def add_noise_to_distances(self, epsilon):
    #     """
    #     Adds Laplace noise to the distances between points in the dataset.
    #     Vectorizes the distance function to simulate a single query.

    #     Parameters:
    #     epsilon (float): The privacy budget for differential privacy.

    #     Returns:
    #     labels (numpy.ndarray): The labels of the data points after adding noise to distances.
    #     """
    #     dp_distance, _ = self._generate_noisy_distances(epsilon)

    #     return self.fit(distance_func=dp_distance)

    def add_noise_to_neighbors(self, epsilon, delta=1e-5):
        """
        Uses Gaussian Noise to add noise to the neighbors of each point in the dataset.

        Parameters:
        epsilon (float): The privacy budget for differential privacy.

        Returns:
        list: Cluster labels for each point (-1 for noise).
        """
        neighbor_matrix = self._generate_noisy_neighbors(epsilon, delta)
        return self.fit(neighbor_matrix)

    def _euclidean_distance(self, i, j):
        """
        Compute the Euclidean distance between two points.

        Parameters:
        i (int): The index of the first point.
        j (int): The index of the second point.

        Returns:
        float: The Euclidean distance between the two points.
        """
        return np.linalg.norm(self.X[i] - self.X[j])

    # def _default_density(self, point_idx):
    #     """
    #     Compute the density of a point by counting the number of data points 
    #     within the radius.

    #     Parameters:
    #     point_idx (int): The point index for which to compute density.

    #     Returns:
    #     int: The number of points within the radius of the point.
    #     """
    #     count = 0
    #     for j in range(self.n_points):
    #         if self._euclidean_distance(point_idx, j) <= self.radius:
    #             count += 1
    #     return count
    
    def _generate_noisy_neighbors(self, epsilon, delta, print_accuracy=False):
        """
        Generate a noisy binary neighbor matrix.
        
        Parameters:
        epsilon (float): The privacy budget for differential privacy.
        
        Returns:
        numpy.ndarray: The noisy binary neighbor matrix.
        """
        # def compute_p(epsilon):
        #     if epsilon > 20:
        #         print(f"[WARNING] Epsilon (ε = {epsilon}) is too large for randomized response. Defaulting to 1.0.")
        #         return 1.0
        #     else:
        #         return np.exp(epsilon) / (1 + np.exp(epsilon))
        
        # num_pairs = self.n_points * (self.n_points - 1) // 2
        # p = compute_p(epsilon / num_pairs)

        # neighbor_matrix = np.zeros((self.n_points, self.n_points), dtype=int)
        # for i in range(self.n_points):
        #     for j in range(self.n_points):
        #         dist = self._euclidean_distance(i, j)
        #         true_neighbor = 1 if dist <= self.radius else 0

        #         flip = np.random.rand() < p
        #         noisy_neighbor = true_neighbor if flip else 1 - true_neighbor

        #         neighbor_matrix[i, j] = noisy_neighbor
        #         neighbor_matrix[j, i] = noisy_neighbor  # for symmetry

        # return neighbor_matrix

        true_matrix = np.zeros((self.n_points, self.n_points), dtype=int)
        for i in range(self.n_points):
            for j in range(self.n_points):
                dist = self._euclidean_distance(i, j)
                true_matrix[i, j] = 1 if dist <= self.radius else 0
                true_matrix[j, i] = true_matrix[i, j]

        sensitivity = np.sqrt(2 * (self.n_points - 1))
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon

        noise = np.random.normal(loc=0, scale=sigma, size=true_matrix.shape)
        noisy = true_matrix.astype(float) + noise
        neighbor_matrix = (noisy > 0.5).astype(int)

        if print_accuracy:
            n = true_matrix.shape[0]

            # create a mask for i<j (upper triangle, excluding diagonal)
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)

            # count matches only where i<j
            matches = (neighbor_matrix[mask] == true_matrix[mask]).sum()
            total_pairs = mask.sum()

            accuracy = matches / total_pairs * 100
            print(f"Neighbor‐matrix accuracy: {accuracy:.2f}%")

        return neighbor_matrix
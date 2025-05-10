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

        self.n = X.shape[0]
        self.dims = X.shape[1]
        
    def fit(self, X=None, distance_func=None):
        """
        Perform DBSCAN clustering.

        Parameters:
        X (numpy.ndarray): The input data for clustering.
        distance_func (callable): A function to compute the distance between points.

        Returns:
        list: Cluster labels for each point (-1 for noise).
        """
        if X is None:
            X = self.X
        if distance_func is None:
            distance_func = self._euclidean_distance

        def region_query(point_idx):
            neighbors = []
            for idx in range(self.n):
                if distance_func(point_idx, idx) <= self.radius:
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
        clusters = [0] * self.n
        cluster_id = 0

        # Iterate through each point
        for point_idx in range(self.n):
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

    
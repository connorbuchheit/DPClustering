# Implement vanilla DBSCAN clustering algorithm
import numpy as np

def dbscan(data, eps, min_samples):
    """
    Implements the DBSCAN clustering algorithm from scratch.

    Parameters:
        data (numpy.ndarray): The dataset as a 2D array.
        eps (float): The maximum distance for two points to be considered neighbors.
        min_samples (int): The minimum number of points required to form a dense region.

    Returns:
        list: Cluster labels for each point in the dataset (-1 indicates noise).
    """
    def region_query(point_idx):
        """Find all neighbors within eps distance of a point."""
        neighbors = []
        for idx, point in enumerate(data):
            if np.linalg.norm(data[point_idx] - point) <= eps:
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
                if len(new_neighbors) >= min_samples:
                    neighbors += new_neighbors
            i += 1

    # Initialize cluster labels (0 = unvisited, -1 = noise)
    clusters = [0] * len(data)
    cluster_id = 0

    # Iterate through each point in the dataset
    for point_idx in range(len(data)):
        if clusters[point_idx] != 0:  # Skip if already visited
            continue
        neighbors = region_query(point_idx)
        if len(neighbors) < min_samples:
            clusters[point_idx] = -1  # Mark as noise
        else:
            cluster_id += 1
            expand_cluster(point_idx, neighbors, cluster_id)

    return clusters

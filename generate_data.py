import numpy as np

def generate_clusters(n_clusters=3, points_per_cluster=100, dim=2, cluster_std=0.5, separation=5):
    X = []
    for i in range(n_clusters):
        center = np.random.uniform(-separation, separation, size=(dim,))
        cluster_points = np.random.randn(points_per_cluster, dim) * cluster_std + center
        X.append(cluster_points)
    return np.vstack(X)
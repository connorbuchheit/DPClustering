from .data import Data

load_csv = Data.load_csv
generate_synthetic_data = Data.generate_synthetic_data
add_noise_to_data = Data.add_noise_to_data
def generate_clusters(n_clusters=3, points_per_cluster=100, dim=2, cluster_std=0.5, separation=5):
    """
    Generates synthetic clusters of data points and returns only the data array.
    """
    X, _ = Data.generate_clusters(
        n_clusters=n_clusters,
        points_per_cluster=points_per_cluster,
        dim=dim,
        cluster_std=cluster_std,
        separation=separation,
    )
    return X
plot_clusters = Data.plot_clusters
clip_rows = Data.clip_rows

__all__ = [
    "load_csv",
    "generate_synthetic_data",
    "add_noise_to_data",
    "generate_clusters",
    "plot_clusters",
    'clip_rows',
]
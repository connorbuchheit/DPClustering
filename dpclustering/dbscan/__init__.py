from .dbscan import DBSCAN

# add_noise_to_densities = DBSCAN.add_noise_to_densities
# add_noise_to_distances = DBSCAN.add_noise_to_distances

# __all__ = [
#     'add_noise_to_densities',
#     'add_noise_to_distances'
# ]

add_noise_to_neighbors = DBSCAN.add_noise_to_neighbors

__all__ = [
    'add_noise_to_neighbors'
]
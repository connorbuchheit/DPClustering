from .data import Data

load_csv = Data.load_csv
generate_synthetic_data = Data.generate_synthetic_data
add_noise_to_data = Data.add_noise_to_data
generate_clusters = Data.generate_clusters
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
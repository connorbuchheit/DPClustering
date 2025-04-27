# DPClustering

This project investigates the application of differential privacy (DP) to common clustering algorithms, specifically k-means and DBSCAN.

### Overview

Our primary goal is to explore the tradeoffs between data privacy and clustering utility (accuracy, interpretability, structure). We aim to understand:

1. How introducing DP noise affects the quality of clustering results.
2. Whether certain algorithm types (centroid-based vs. density-based) are inherently more robust or suitable for DP.
3. The impact of applying DP noise at different stages of the clustering process *(eg., input data, and intermediate calculations like centroids/densities)*.

### Implementation Status

| Feature                   | Description                                                                                                                    |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| Baseline Implementations  | Non-private versions of k-means (`k_means.py`) and DBSCAN (`dbscan.py`) implemented from scratch                              |
| Synthetic Data Generation | Utility function (`generate_data.py`) to create simple clustered datasets for testing                                          |
| Basic Visualization       | `main.py` script generates synthetic data, runs both non-private algorithms, and visualizes results using Matplotlib           |

### Usage (Current Baseline)

To run the current non-private implementations and see the visualization, ensure numpy and matplotlib are installed, then run the `main.py` script:

```bash
pip install numpy matplotlib
```

Then:
- clone the repository
- navigate to the project directory in your terminal
- run the main script:

```bash
python main.py
```

*this will generate synthetic data, run k-means and DBSCAN, and display a plot comparing their outputs.*
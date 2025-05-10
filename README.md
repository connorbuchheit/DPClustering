# DPClustering

This project investigates the application of differential privacy (DP) to common clustering algorithms, specifically k-means and DBSCAN.

## 1. Overview

Our primary goal is to explore the tradeoffs between data privacy and clustering utility (accuracy, interpretability, structure). We aim to understand:

1. How introducing DP noise affects the quality of clustering results.
2. Whether certain algorithm types (centroid-based vs. density-based) are inherently more robust or suitable for DP.
3. The impact of applying DP noise at different stages of the clustering process *(eg., input data, and intermediate calculations like centroids/densities)*.

## 2. Implementation Status

| Feature                   | Description                                                                                                                    |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| Baseline Implementations  | Non-private versions of k-means (`k_means.py`) and DBSCAN (`dbscan.py`) implemented from scratch                              |
| Synthetic Data Generation | Utility function (`generate_data.py`) to create simple clustered datasets for testing                                          |
| Basic Visualization       | `main.py` script generates synthetic data, runs both non-private algorithms, and visualizes results using Matplotlib           |
| DP Mechanisms             | Implementations for adding noise to data, centroids (k-means), and neighbor checks (DBSCAN)                                    |
| Evaluation Scripts        | Scripts in `tests/` for running experiments on synthetic and real (insurance) data, calculating metrics (ARI, NMI), and plotting |

## 3. Usage

To run the various scripts and experiments in this project:

1. *clone the repository:*

    ```bash
    git clone https://github.com/connorbuchheit/DPClustering
    cd ./DPClustering
    ```

2. *install required packages:*

    ```bash
    pip install -r requirements.txt
    ```

3. *use the Makefile targets:* The `Makefile` provides convenient commands to run different parts of the project. The `PYTHONPATH` is set automatically. (you will get an error if you try to run the scripts directly without the Makefile, but if you set the PYTHONPATH manually it should also work, we do recommend the Makefile however!)

### (a) Run the main synthetic data example

```bash
make run-main
```

This executes `main.py`.

### (b) Run insurance-specific experiments and generate plots

```bash
make run-insurance-plots
```

This executes `tests/insurance.py`, which runs experiments specifically on the insurance dataset and generates comparison plots (like those potentially used in the report).

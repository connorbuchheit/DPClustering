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

### Usage

To run the current non-private implementations and see the visualization:

1. clone the repository
2. install the required packages
3. run the main script:

```bash
pip install -r requirements.txt
python main.py
```

*this will generate synthetic data, run k-means and DBSCAN, and display a plot comparing their outputs.*

#### Tests

To run the clustering tests located in the `tests/` folder, use the provided `Makefile`:

```bash
make test
```

This sets the `PYTHONPATH` automatically. Alternatively, you can still run:

```bash
python tests/test.py
```

But ensure you have set the `PYTHONPATH` correctly to include the current directory, or you will encounter a `ModuleNotFoundError`:

##### Common Error

If you encounter the following:

```bash
python tests/test.py
Traceback (most recent call last):
  File "/Users/bqr/class/208/DPClustering/tests/test.py", line 1, in <module>
    import dpclustering as dpc
ModuleNotFoundError: No module named 'dpclustering'
```

You would need to set the `PYTHONPATH` to the current directory so Python can find the `dpclustering` module. A simple way to do this is by running:

```bash
export PYTHONPATH=$PWD #this initialises a new environment variable $PYTHONPATH, and gives it value of the current working directory ($PWD == print working directory)

python tests/test.py
--------------------------
ARI (add_noise_to_densi...
```
# appends parent directory to access dpclustering folder
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dpclustering as dpc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.datasets import make_moons, make_circles, make_blobs

def test_clusters(X, k, b, epsilon, max_iter, plot=True):
    # ==================================
    # epsilon-dp data
    # ==================================
    noisy_data = dpc.data.add_noise_to_data(X, epsilon, sensitivity=1, b=b)

    # ==================================
    # private clustering
    # =================================

    # private k-means clustering
    kmeans = dpc.kmeans.KMeans(X, k, b, max_iter)
    centroids, kmeans_labels = kmeans.fit()
    
    # private dbscan clustering
    dbscan = dpc.dbscan.DBSCAN(X, radius=0.15, min_samples=9, b=b, max_iter=max_iter)
    dbscan_labels = dbscan.fit()

    # ==================================
    # epsilon-dp k-means clustering
    # ==================================

    # post-processing
    dp_centroids1, dp_kmeans_labels1 = kmeans.add_noise_to_centroids(epsilon)

    # pre-processing
    kmeans.X = noisy_data
    dp_centroids2, dp_kmeans_labels2 = kmeans.fit()

    # ==================================
    # epsilon-dp dbscan clustering
    # ==================================

    # intermediate processing
    dp_dbscan_labels1 = dbscan.add_noise_to_densities(epsilon)
    dp_dbscan_labels2 = dbscan.add_noise_to_distances(epsilon)

    # pre-processing
    dbscan.X = noisy_data
    dp_dbscan_labels3 = dbscan.fit()

    # ==================================
    # ARI scores
    # ==================================
    ari_scores = {
        "dp_kmeans_labels1": ari(kmeans_labels, dp_kmeans_labels1),
        "dp_kmeans_labels2": ari(kmeans_labels, dp_kmeans_labels2),
        "dp_dbscan_labels1": ari(dbscan_labels, dp_dbscan_labels1),
        "dp_dbscan_labels2": ari(dbscan_labels, dp_dbscan_labels2),
        "dp_dbscan_labels3": ari(dbscan_labels, dp_dbscan_labels3)
    }

    print("--------------------------")
    print("ARI (k-means + centroid noise):", ari_scores["dp_kmeans_labels1"])
    print("ARI (k-means + dp dataset):", ari_scores["dp_kmeans_labels2"])
    print("---------------------------")
    print("ARI (dbscan + density noise):", ari_scores["dp_dbscan_labels1"])
    print("ARI (dbscan + distance noise):", ari_scores["dp_dbscan_labels2"])
    print("ARI (dbscan + dp dataset):", ari_scores["dp_dbscan_labels3"])
    print("---------------------------")

    # ==================================
    # plot clusters
    # ==================================
    if plot:
        dpc.data.plot_clusters(X, kmeans_labels, title="Original KMeans")
        dpc.data.plot_clusters(X, dp_kmeans_labels1, title="KMeans + Centroid Noise")
        dpc.data.plot_clusters(X, dp_kmeans_labels2, title="KMeans + DP Dataset")

        dpc.data.plot_clusters(X, dbscan_labels, title="Original DBSCAN")
        dpc.data.plot_clusters(X, dp_dbscan_labels1, title="DBSCAN + Density Noise")
        dpc.data.plot_clusters(X, dp_dbscan_labels2, title="DBSCAN + Distance Noise")
        dpc.data.plot_clusters(X, dp_dbscan_labels3, title="DBSCAN + DP Dataset")

    return ari_scores

def test_moons():
    # Generate synthetic data using make_moons
    X, _ = make_moons(n_samples=1000, noise=0.1)
    k = 2
    b = 5
    epsilon = 0.1
    max_iter = 2000

    test_clusters(X, k, b, epsilon, max_iter)

test_moons()
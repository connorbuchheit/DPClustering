 

import dpclustering as dpc
import numpy as np
import matplotlib.pyplot as plt

k = 3
b = 5
data = dpc.data.generate_clusters(points_per_cluster=100, cluster_std=0.5, n_clusters=k, separation=b)

epsilon = 10.0
max_iter = 100

def test_kmeans():
    # private k-means clustering
    kmeans = dpc.kmeans.KMeans(data, k, b, max_iter)
    centroids, labels = kmeans.fit()
    
    # testing all dp clustering methods
    dp_centroids1, dp_labels1 = kmeans.add_noise_to_centroids(epsilon)

    noisy_data = dpc.data.add_noise_to_data(data, epsilon, sensitivity=1, b=b)
    kmeans.X = noisy_data
    dp_centroids2, dp_labels2 = kmeans.fit()

    from sklearn.metrics import adjusted_rand_score
    print("ARI (add_noise_to_centroids):", adjusted_rand_score(labels, dp_labels1))
    print("ARI (add_noise_to_data):", adjusted_rand_score(labels, dp_labels2))

    # dpc.data.plot_clusters(data, labels, title="Original")
    # dpc.data.plot_clusters(data, dp_labels1, title="Add Noise to Centroids")
    # dpc.data.plot_clusters(data, dp_labels2, title="Add Noise to Data")

def test_dbscan():
    # private dbscan clustering
    dbscan = dpc.dbscan.DBSCAN(data, radius=0.5, min_samples=5, b=b, max_iter=2000)
    labels = dbscan.fit()
    
    # testing all dp clustering methods
    dp_labels1 = dbscan.add_noise_to_densities(epsilon)
    dp_labels2 = dbscan.add_noise_to_distances(epsilon)

    noisy_data = dpc.data.add_noise_to_data(data, epsilon, sensitivity=1, b=b)
    dbscan.X = noisy_data
    dp_labels3 = dbscan.fit()

    from sklearn.metrics import adjusted_rand_score

    print("--------------------------")
    print("ARI (add_noise_to_densities):", adjusted_rand_score(labels, dp_labels1))
    print("ARI (add_noise_to_distances):", adjusted_rand_score(labels, dp_labels2))
    print("ARI (add_noise_to_data):", adjusted_rand_score(labels, dp_labels3))

    dpc.data.plot_clusters(data, labels, title="Original")
    dpc.data.plot_clusters(data, dp_labels1, title="Add Noise to Densities")
    dpc.data.plot_clusters(data, dp_labels2, title="Add Noise to Distances")
    dpc.data.plot_clusters(data, dp_labels3, title="Add Noise to Data")
    


# test_kmeans()
test_dbscan()
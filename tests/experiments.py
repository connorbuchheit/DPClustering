# appends parent directory to access dpclustering folder
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dpclustering as dpc
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.datasets import make_moons, make_circles, make_blobs

def test_clusters(X, k, b, epsilon, max_iter, plot=True, true_labels=None, silent=True):
    if not silent:
        print("Settings:")
        print("k:", k)
        print("b:", b)
        print("epsilon:", epsilon)
        print("max_iter:", max_iter)

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
    dbscan = dpc.dbscan.DBSCAN(X, radius=0.1, min_samples=6, b=b, max_iter=max_iter)
    dbscan_labels = dbscan.fit()

    if true_labels is None:
        true_labels_kmeans, true_labels_dbscan = kmeans_labels, dbscan_labels
    else:
        true_labels_kmeans, true_labels_dbscan = true_labels, true_labels

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
    dp_dbscan_labels1 = dbscan.add_noise_to_neighbors(epsilon)

    # pre-processing
    dbscan.X = noisy_data
    dp_dbscan_labels2 = dbscan.fit()

    # ==================================
    # ARI scores
    # ==================================
    ari_scores = {
        "original_kmeans": ari(true_labels_kmeans, kmeans_labels),
        "original_dbscan": ari(true_labels_dbscan, dbscan_labels),
        "dp_kmeans_labels1": ari(true_labels_kmeans, dp_kmeans_labels1),
        "dp_kmeans_labels2": ari(true_labels_kmeans, dp_kmeans_labels2),
        "dp_dbscan_labels1": ari(true_labels_dbscan, dp_dbscan_labels1),
        "dp_dbscan_labels2": ari(true_labels_dbscan, dp_dbscan_labels2),
    }

    for key, value in ari_scores.items():
        ari_scores[key] = round(value, 4)

    if not silent:
        print("--------------------------")
        print("ARI (k-means + centroid noise):", ari_scores["dp_kmeans_labels1"])
        print("ARI (k-means + dp dataset):", ari_scores["dp_kmeans_labels2"])
        print("---------------------------")
        print("ARI (dbscan + neighbor noise):", ari_scores["dp_dbscan_labels1"])
        print("ARI (dbscan + dp dataset):", ari_scores["dp_dbscan_labels2"])
        print("---------------------------")

    # ==================================
    # NMI scores
    # ==================================
    nmi_scores = {
        "original_kmeans": nmi(true_labels_kmeans, kmeans_labels),
        "original_dbscan": nmi(true_labels_dbscan, dbscan_labels),
        "dp_kmeans_labels1": nmi(true_labels_kmeans, dp_kmeans_labels1),
        "dp_kmeans_labels2": nmi(true_labels_kmeans, dp_kmeans_labels2),
        "dp_dbscan_labels1": nmi(true_labels_dbscan, dp_dbscan_labels1),
        "dp_dbscan_labels2": nmi(true_labels_dbscan, dp_dbscan_labels2),
    }

    for key, value in nmi_scores.items():
        nmi_scores[key] = round(value, 4)

    if not silent:
        print("NMI (k-means + centroid noise):", nmi_scores["dp_kmeans_labels1"])
        print("NMI (k-means + dp dataset):", nmi_scores["dp_kmeans_labels2"])
        print("---------------------------")
        print("NMI (dbscan + neighbor noise):", nmi_scores["dp_dbscan_labels1"])
        print("NMI (dbscan + dp dataset):", nmi_scores["dp_dbscan_labels2"])
        print("---------------------------")

    # ==================================
    # plot clusters
    # ==================================
    if plot:
        if true_labels is not None:
            dpc.data.plot_clusters(X, true_labels, title="True Labels")

        dpc.data.plot_clusters(X, kmeans_labels, title="Original KMeans")
        dpc.data.plot_clusters(X, dp_kmeans_labels1, title="KMeans + Centroid Noise")
        dpc.data.plot_clusters(X, dp_kmeans_labels2, title="KMeans + DP Dataset")

        dpc.data.plot_clusters(X, dbscan_labels, title="Original DBSCAN")
        dpc.data.plot_clusters(X, dp_dbscan_labels1, title="DBSCAN + Neighbor Noise")
        dpc.data.plot_clusters(X, dp_dbscan_labels2, title="DBSCAN + DP Dataset")

    return ari_scores, nmi_scores

def run_experiments(epsilon_values, n_trials=5):
    results = {
        'moons': {'epsilon': []},
        'circles': {'epsilon': []},
        'blobs': {'epsilon': []},
        'synthetic': {'epsilon': []},
    }

    variants = ['original_kmeans', 'original_dbscan', 'dp_kmeans_labels1', 'dp_kmeans_labels2', 'dp_dbscan_labels1', 'dp_dbscan_labels2']
    metrics = ['ari', 'nmi']

    for dataset in results.keys():
        for variant in variants:
            for metric in metrics:
                results[dataset][f'{variant}_{metric}'] = []

    total_steps = len(epsilon_values) * len(results) * n_trials
    pbar = tqdm(total=total_steps, desc="Running experiments", ncols=100)

    for eps in epsilon_values:
        for dataset_name, dataset_func, k in [
            ('moons', lambda: make_moons(n_samples=1000, noise=0.1), 2),
            ('circles', lambda: make_circles(n_samples=1000, factor=0.3, noise=0.1), 2),
            ('blobs', lambda: make_blobs(n_samples=1000, centers=3, cluster_std=.8), 3),
            ('synthetic', lambda: dpc.data.generate_clusters(n_clusters=4, points_per_cluster=300, cluster_std=2, separation=5), 4),
        ]:
            all_ari = {variant: [] for variant in variants}
            all_nmi = {variant: [] for variant in variants}

            for _ in range(n_trials):
                X, true_labels = dataset_func()
                ari_scores, nmi_scores = test_clusters(X, k=k, b=5, epsilon=eps, max_iter=2000, plot=False, true_labels=true_labels)

                for variant in variants:
                    all_ari[variant].append(ari_scores[variant])
                    all_nmi[variant].append(nmi_scores[variant])

                pbar.update(1)  # update progress bar after each trial

            results[dataset_name]['epsilon'].append(eps)

            for variant in variants:
                avg_ari = np.mean(all_ari[variant])
                avg_nmi = np.mean(all_nmi[variant])

                results[dataset_name][f'{variant}_ari'].append(avg_ari)
                results[dataset_name][f'{variant}_nmi'].append(avg_nmi)

    pbar.close()
    return results


def plot_all_results(results, metric='ari'):
    datasets = ['moons', 'circles', 'blobs', 'synthetic']
    variants = ['original_kmeans', 'original_dbscan', 'dp_kmeans_labels1', 'dp_kmeans_labels2', 'dp_dbscan_labels1', 'dp_dbscan_labels2']
    variant_labels = {
        'original_kmeans': 'Original KMeans',
        'original_dbscan': 'Original DBSCAN',
        'dp_kmeans_labels1': 'KMeans + Centroid Noise',
        'dp_kmeans_labels2': 'KMeans + DP Dataset',
        'dp_dbscan_labels1': 'DBSCAN + Neighbor Noise',
        'dp_dbscan_labels2': 'DBSCAN + DP Dataset',
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        for variant in variants:
            ax.plot(results[dataset]['epsilon'], results[dataset][f'{variant}_{metric}'], label=variant_labels[variant])

        ax.set_title(f'{dataset.capitalize()} - {metric.upper()}')
        ax.set_xlabel('Epsilon')
        ax.set_ylabel(metric.upper())
        ax.grid(True)
        ax.legend(fontsize='small')

    plt.tight_layout()
    plt.show()


epsilon_values = [0.1, 0.5, 1, 2, 5, 7.5, 10, 15, 20, 50]
results = run_experiments(epsilon_values)

plot_all_results(results, metric='ari')
plot_all_results(results, metric='nmi')

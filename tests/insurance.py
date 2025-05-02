import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
import seaborn as sns

import dpclustering as dpc

EPSILON = 20
B = 3 # clipping

K = 4 # Number of clusters (for k-means)
RADIUS = 3 # (for DBSCAN)
MIN_SAMPLES = 6 # (for DBSCAN)
MAX_ITER = 2000 

DELTA = 1e-5 # Used for any approx-DP algorithms
PHI2 = 1/1000 # Used only for private k-means. phi^2 must be set <=1/1000

DIMS = 2 # number of dimensions to plot graph
SEED = None 

np.random.seed(SEED)

def main():
    X = np.clip(scale_data(insurance_to_numpy()), -B, B)
    X_noisy = noise_data(X, EPSILON, sensitivity=2 * B, b=B)
    X_low_dim = reduce_dimensions(X, DIMS)

    np_km_centroids, np_km_labels = non_private_kmeans(X, K)
    np_db_labels = non_private_dbscan(X, radius=3, min_samples=6)

    dpc_km_centroids, dpc_km_labels = dpc_dp_kmeans(X, K, B, EPSILON)
    dpc_km_centroids2, dpc_km_labels2 = dpc_private_kmeans(X, K, B, EPSILON, DELTA, PHI2)

    dpc_db_labels = dpc_dp_dbscan(X, radius=RADIUS, min_samples=MIN_SAMPLES, b=B, epsilon=EPSILON)

    dpdata_km_centroids, dpdata_km_labels = dpc_dp_kmeans(X_noisy, K, B, EPSILON)
    dpdata_db_labels = dpc_dp_dbscan(X_noisy, radius=RADIUS, min_samples=MIN_SAMPLES, b=B, epsilon=EPSILON)

    print(f"EPSILON = {EPSILON}")

    ari_scores(np_km_labels, [{"Noisy Centroids K-Means": dpc_km_labels}, {"Private K-Means": dpc_km_labels2}, {"Noisy Data K-Means": dpdata_km_labels}])
    ari_scores(np_db_labels, [{"DPC DBSCAN": dpc_db_labels}, {"Noisy Data DBSCAN": dpdata_db_labels}])
    nmi_scores(np_km_labels, [{"Noisy Centroids K-Means": dpc_km_labels}, {"Private K-Means": dpc_km_labels2}, {"Noisy Data K-Means": dpdata_km_labels}])
    nmi_scores(np_db_labels, [{"DPC DBSCAN": dpc_db_labels}, {"Noisy Data DBSCAN": dpdata_db_labels}])

    normalized_distances(np_km_centroids, [{"Noisy Centroids K-Means": dpc_km_centroids}, {"Private K-Means": dpc_km_centroids2}, {"Noisy Data K-Means": dpdata_km_centroids}])

    dpc.data.plot_clusters(X_low_dim, np_km_labels, title=f"Non-private KMeans", dims=DIMS)
    dpc.data.plot_clusters(X_low_dim, np_db_labels, title=f"Non-private DBSCAN", dims=DIMS)
    dpc.data.plot_clusters(X_low_dim, dpc_db_labels, title=f"DP DBSCAN (epsilon = {EPSILON})", dims=DIMS)
    dpc.data.plot_clusters(X_low_dim, dpdata_km_labels, title=f"KMeans (noisy data, epsilon = {EPSILON})", dims=DIMS)
    dpc.data.plot_clusters(X_low_dim, dpdata_db_labels, title=f"DBSCAN (noisy data, epsilon = {EPSILON})", dims=DIMS)
    dpc.data.plot_clusters(X_low_dim, dpc_km_labels2, title=f"Private KMeans (epsilon = {EPSILON}, delta = {DELTA})", dims=DIMS)


def boxwhiskers():
    EPSILONS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    NUM_TRIALS = 10

    ari_results = {algo: {eps: [] for eps in EPSILONS} for algo in ["Noisy Centroids KMeans", "Private KMeans", "Noisy Data KMeans"]}
    nmi_results = {algo: {eps: [] for eps in EPSILONS} for algo in ["Noisy Centroids KMeans", "Private KMeans", "Noisy Data KMeans"]}

    for eps in EPSILONS:
        for _ in range(NUM_TRIALS):
            X = np.clip(scale_data(insurance_to_numpy()), -B, B)
            X_noisy = noise_data(X, eps, sensitivity=2 * B, b=B)

            np_km_centroids, np_km_labels = non_private_kmeans(X, K)

            dpc_km_centroids, dpc_km_labels = dpc_dp_kmeans(X, K, B, eps)
            dpc_km_centroids2, dpc_km_labels2 = dpc_private_kmeans(X, K, B, eps, DELTA, PHI2)
            dpdata_km_centroids, dpdata_km_labels = dpc_dp_kmeans(X_noisy, K, B, eps)

            # ARI/NMI scores
            ari_results["Noisy Centroids KMeans"][eps].append(ari(np_km_labels, dpc_km_labels))
            ari_results["Private KMeans"][eps].append(ari(np_km_labels, dpc_km_labels2))
            ari_results["Noisy Data KMeans"][eps].append(ari(np_km_labels, dpdata_km_labels))

            nmi_results["Noisy Centroids KMeans"][eps].append(nmi(np_km_labels, dpc_km_labels))
            nmi_results["Private KMeans"][eps].append(nmi(np_km_labels, dpc_km_labels2))
            nmi_results["Noisy Data KMeans"][eps].append(nmi(np_km_labels, dpdata_km_labels))


    def flatten_results(results_dict, metric_name):
        records = []
        for algo, eps_dict in results_dict.items():
            for eps, scores in eps_dict.items():
                for score in scores:
                    records.append({
                        "Epsilon": eps,
                        "Algorithm": algo,
                        metric_name: score
                    })
        return pd.DataFrame(records)

    # Create DataFrames
    df_ari = flatten_results(ari_results, "ARI")
    df_nmi = flatten_results(nmi_results, "NMI")

    # Plot ARI boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Epsilon", y="ARI", hue="Algorithm", data=df_ari)
    plt.title("ARI Distribution by Epsilon and Algorithm")
    plt.legend(title="Algorithm", loc="best")
    plt.show()

    # Plot NMI boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Epsilon", y="NMI", hue="Algorithm", data=df_nmi)
    plt.title("NMI Distribution by Epsilon and Algorithm")
    plt.legend(title="Algorithm", loc="best")
    plt.show()



def insurance_to_numpy():
    df = dpc.data.load_csv("csv/insurance.csv")

    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})

    region_encoded = pd.get_dummies(df["region"], prefix="region")
    df = pd.concat([df.drop("region", axis=1), region_encoded], axis=1)

    features = ["age", 'sex', 'bmi', 'children', 'smoker'] + list(region_encoded.columns)
    return df[features].values

def scale_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def reduce_dimensions(X, dims):
    pca = PCA(n_components=dims)
    return pca.fit_transform(X)

def noise_data(X, epsilon, sensitivity, b):
    return dpc.data.add_noise_to_data(X, epsilon, sensitivity=sensitivity, b=b)

def non_private_kmeans(X, k):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    return centroids, labels

def non_private_dbscan(X, radius, min_samples):
    dbscan = DBSCAN(eps=radius, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return labels

def dpc_dp_kmeans(X, k, b, epsilon):
    dp_kmeans = dpc.kmeans.KMeans(X, k=k, b=b, max_iter=2000)
    dp_kmeans.fit()
    return dp_kmeans.add_noise_to_centroids(epsilon=epsilon)

def dpc_private_kmeans(X, k, b, epsilon, delta, phi2):
    dp_kmeans = dpc.kmeans.KMeans(X, k=k, b=b, max_iter=2000)
    return dp_kmeans.private_kmeans(epsilon=epsilon, delta=delta, phi2=phi2)

def dpc_dp_dbscan(X, radius, min_samples, b, epsilon):
    dp_dbscan = dpc.dbscan.DBSCAN(X, radius=radius, min_samples=min_samples, b=b, max_iter=2000)
    return dp_dbscan.add_noise_to_neighbors(epsilon=epsilon)

def normalized_distance(c1, c2):
    distance = np.linalg.norm(c1 - c2)
    max_distance = 2 * B * np.sqrt(DIMS)
    return distance / max_distance

def normalized_distances(c1, c2_dict_list):
    for c2 in c2_dict_list:
        for key, value in c2.items():
            distance = normalized_distance(c1, value)
            print(f"Normalized distance between centroids and {key}: {distance:.4f}")

def ari_scores(true_labels, dpc_labels_dict_list):
    for dpc_labels in dpc_labels_dict_list:
        for key, value in dpc_labels.items():
            score = ari(true_labels, value)
            print(f"ARI between true labels and {key}: {score:.4f}")

def nmi_scores(true_labels, dpc_labels_dict_list):
    for dpc_labels in dpc_labels_dict_list:
        for key, value in dpc_labels.items():
            score = nmi(true_labels, value)
            print(f"NMI between true labels and {key}: {score:.4f}")

# main()
boxwhiskers()
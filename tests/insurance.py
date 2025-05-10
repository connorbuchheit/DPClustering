import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA 
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.cluster import KMeans
from diffprivlib.models import KMeans as dplibKMeans
import seaborn as sns
from tqdm import tqdm

import dpclustering as dpc

EPSILON = 1.0
B = 3 # clipping

K = 4 # Number of clusters (for k-means)
RADIUS = 3 # (for DBSCAN)
MIN_SAMPLES = 6 # (for DBSCAN)
MAX_ITER = 2000 

kNN = 5 # number of nearest neighbors for density peaks clustering

DIMS = 2 # number of dimensions to plot graph
SEED = None 

np.random.seed(SEED)

def main():
    """
    Generates box and whisker plots for ARI, NMI, and quantization loss for 
    different methods of clustering.
    """

    epsilons = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    n_trials = 10

    results = {
        "Epsilon": [],
        "Method": [],
        "ARI": [],
        "NMI": [],
        "QuantizationLoss": [],
        "Labels": [],
        "Centers": [],
    }

    X = insurance_to_numpy()
    for epsilon in tqdm(epsilons, desc="Epsilons"):
        for trial in tqdm(range(n_trials), desc=f"Trials for ε={epsilon}", leave=False):
            # np.random.seed(trial)

            noisy_X = dpc.data.add_noise_to_data(
                X, epsilon=epsilon, sensitivity=2 * B * X.shape[1], b=B
            )
            noisy_X = np.clip(noisy_X, -B, B)

            double_noisy_X = dpc.data.add_noise_to_data(
                X, epsilon=epsilon/2, sensitivity=2 * B * X.shape[1], b=B
            )
            double_noisy_X = np.clip(double_noisy_X, -B, B)

            # True KMeans
            true_kmeans = KMeans(n_clusters=K, random_state=SEED)
            true_kmeans.fit(X)
            true_centroids = true_kmeans.cluster_centers_
            y = true_kmeans.predict(X)

            # DPC KMeans
            dpc_kmeans = dpc.kmeans.KMeans(X, k=K, b=B, max_iter=MAX_ITER)
            balcan_centers = dpc_kmeans.balcanetal_fit(epsilon=epsilon)
            dpdata_centers, _ = dpc_kmeans.fit(noisy_X)

            # diffprivlib KMeans
            dplib_kmeans = dplibKMeans(n_clusters=K, epsilon=epsilon, bounds=(-B, B), random_state=SEED)
            dplib_kmeans.fit(X)
            dplib_centers = dplib_kmeans.cluster_centers_

            dplib_kmeans = dplibKMeans(n_clusters=K, epsilon=epsilon/2, bounds=(-B, B), random_state=SEED)
            dplib_kmeans.fit(double_noisy_X)
            dplib_dpdata_centers = dplib_kmeans.cluster_centers_

            for method, centers in [
                ("DPC KMeans (DPData)", dpdata_centers),
                ("DPC KMeans (Balcan et al.)", balcan_centers),
                ("diffprivlib KMeans", dplib_centers),
                ("diffprivlib KMeans (DPData)", dplib_dpdata_centers),
            ]:
                labels = dpc.data.predict(X, centers)
                centers, labels = canonicalize_labels(X, centers)
                ari_score = ari(y, labels)
                nmi_score = nmi(y, labels)
                loss = dpc.data.quantization_loss(X, centers)
                X_low_dims = reduce_dimensions(X, dims=DIMS)
                

                results["Epsilon"].append(epsilon)
                results["Method"].append(method)
                results["ARI"].append(ari_score)
                results["NMI"].append(nmi_score)
                results["QuantizationLoss"].append(loss)
                results["Labels"].append(labels)
                results["Centers"].append(centers)
                    
    df = pd.DataFrame(results)

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(x="Epsilon", y="ARI", hue="Method", data=df)
    plt.title("ARI vs Epsilon")

    plt.subplot(1, 2, 2)
    sns.boxplot(x="Epsilon", y="NMI", hue="Method", data=df)
    plt.title("NMI vs Epsilon")
    plt.tight_layout()
    plt.legend(loc='upper left', fontsize='small')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Epsilon", y="QuantizationLoss", hue="Method", data=df)
    plt.title("Quantization Loss vs Epsilon")
    plt.tight_layout()
    plt.show()

    X_low_dims = reduce_dimensions(X, dims=DIMS)

    df_eps1 = df[df["Epsilon"] == 1.0]

    methods = df_eps1["Method"].unique()
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4), squeeze=False)

    for idx, method in enumerate(methods):
        row = df_eps1[df_eps1["Method"] == method].iloc[0]
        centers = row["Centers"]
        labels = row["Labels"]
        ax = axes[0, idx]
        dpc.data.plot_clusters(
            X_low_dims,
            labels,
            title=method + f" (ε = 1.0)",
            dims=DIMS,
            show=False,
            centers=centers,
            ax=ax
        )

    plt.tight_layout()
    plt.show()

def insurance_to_numpy():
    df = dpc.data.load_csv("csv/insurance.csv")

    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})

    region_encoded = pd.get_dummies(df["region"], prefix="region")
    df = pd.concat([df.drop("region", axis=1), region_encoded], axis=1)

    features = ["age", 'sex', 'bmi', 'children', 'smoker'] + list(region_encoded.columns)
    
    return np.clip(scale_data(df[features].values), -B, B)

def scale_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def reduce_dimensions(X, dims):
    pca = PCA(n_components=dims)
    return pca.fit_transform(X)

def quantization_loss(X, centers):
    dists = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    min_sq_dists = np.min(dists **2, axis=1)
    return np.mean(min_sq_dists)

def canonicalize_labels(X, centers):
    from scipy.spatial.distance import cdist

    sorted_indices = np.argsort(centers[:, 0])
    sorted_centers = centers[sorted_indices]

    distances = cdist(X, sorted_centers)
    new_labels = np.argmin(distances, axis=1)

    return sorted_centers, new_labels




main()
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
    X = insurance_to_numpy()
    noisy_X = dpc.data.add_noise_to_data(X, epsilon=EPSILON, sensitivity=2 * B, b=B)
    noisy_X = np.clip(noisy_X, -B, B)

    # True KMeans clustering
    true_kmeans = KMeans(n_clusters=K, random_state=SEED)
    true_kmeans.fit(X)
    true_centroids = true_kmeans.cluster_centers_

    # DPC KMeans clustering
    dpc_kmeans = dpc.kmeans.KMeans(X, k=K, b=B, max_iter=MAX_ITER)
    dpc_balcanetal_centroids = dpc_kmeans.balcanetal_fit(epsilon=EPSILON)
    dpc_dpdata_centroids, _ = dpc_kmeans.fit(noisy_X)

    # Losses
    loss_true = quantization_loss(X, true_centroids)
    loss_dpc_balcanetal = quantization_loss(X, dpc_balcanetal_centroids)
    loss_dpc_dpdata = quantization_loss(X, dpc_dpdata_centroids)

    print(f"True KMeans Loss: {loss_true:.4f}")
    print(f"DPC KMeans Balcan et al Loss: {loss_dpc_balcanetal:.4f}")
    print(f"DPC KMeans DPData Loss: {loss_dpc_dpdata:.4f}")

    

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

main()
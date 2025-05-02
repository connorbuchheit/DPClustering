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

import dpclustering as dpc

EPSILON = 10 # Epsilon must be set <=20.
SEED = None
DIMS = 2

np.random.seed(SEED)

df = dpc.data.load_csv("csv/insurance.csv")
print("Number of rows:", df.shape[0])

df["sex"] = df["sex"].map({"male": 0, "female": 1})
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})

region_encoded = pd.get_dummies(df["region"], prefix="region")
df = pd.concat([df.drop("region", axis=1), region_encoded], axis=1)

features = ["age", 'sex', 'bmi', 'children', 'smoker'] + list(region_encoded.columns)
X = df[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=DIMS)
X_low_dim = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=4)
kmeans_labels = kmeans.fit_predict(X_scaled)

dbscan = DBSCAN(eps=3, min_samples=6)
dbscan_labels = dbscan.fit_predict(X_scaled)

dp_kmeans = dpc.kmeans.KMeans(X_scaled, k=4, b=3, max_iter=2000)
dp_kmeans.fit()
dp_centroids, dp_kmeans_labels = dp_kmeans.add_noise_to_centroids(epsilon=EPSILON)

pkm_centroids, pkm_labels = dp_kmeans.private_kmeans(epsilon=EPSILON, delta=1e-5, phi2=1/1000)

dp_dbscan = dpc.dbscan.DBSCAN(X_scaled, radius=3, min_samples=6, b=3, max_iter=2000)
dp_dbscan_labels = dp_dbscan.add_noise_to_neighbors(epsilon=EPSILON)

X_scaled_clipped = dpc.data.clip_rows(X_scaled, b=3)
X_scaled_noisy = dpc.data.add_noise_to_data(X_scaled, epsilon=EPSILON, sensitivity=1, b=3)
dpdata_kmeans_labels = kmeans.fit_predict(X_scaled_noisy)

dp_dbscan.X = X_scaled_noisy
dpdata_dbscan_labels = dbscan.fit_predict(X_scaled_noisy)

ari_kmeans        = ari(kmeans_labels, dp_kmeans_labels)
ari_dpdata_kmeans = ari(kmeans_labels, dpdata_kmeans_labels)
ari_pkm           = ari(kmeans_labels, pkm_labels)
ari_dbscan        = ari(dbscan_labels, dp_dbscan_labels)
ari_dpdata_dbscan = ari(dbscan_labels, dpdata_dbscan_labels)
nmi_kmeans        = nmi(kmeans_labels, dp_kmeans_labels)
nmi_dpdata_kmeans = nmi(kmeans_labels, dpdata_kmeans_labels)
nmi_pkm           = nmi(kmeans_labels, pkm_labels)
nmi_dbscan        = nmi(dbscan_labels, dp_dbscan_labels)
nmi_dpdata_dbscan = nmi(dbscan_labels, dpdata_dbscan_labels)

print("Epsilon =", EPSILON)
print(f"ARI between KMeans and DPC KMeans: {ari_kmeans:.4f}")
print(f"ARI between DBSCAN and DPC DBSCAN: {ari_dbscan:.4f}")
print(f"ARI between KMeans and Private KMeans: {ari_pkm:.4f}")
print(f"ARI between KMeans and KMeans (noisy data): {ari_dpdata_kmeans:.4f}")
print(f"ARI between DBSCAN and DBSCAN (noisy data): {ari_dpdata_dbscan:.4f}")
print(f"NMI between KMeans and DPC KMeans: {nmi_kmeans:.4f}")
print(f"NMI between DBSCAN and DPC DBSCAN: {nmi_dbscan:.4f}")
print(f"NMI between KMeans and Private KMeans: {nmi_pkm:.4f}")
print(f"NMI between KMeans and KMeans (noisy data): {nmi_dpdata_kmeans:.4f}")
print(f"NMI between DBSCAN and DBSCAN (noisy data): {nmi_dpdata_dbscan:.4f}")

dpc.data.plot_clusters(X_low_dim, kmeans_labels, "Standard KMeans", dims=DIMS)
dpc.data.plot_clusters(X_low_dim, dbscan_labels, "Standard DBSCAN", dims=DIMS)
dpc.data.plot_clusters(X_low_dim, dp_kmeans_labels, f"DPC KMeans (epsilon={EPSILON})", dims=DIMS)
dpc.data.plot_clusters(X_low_dim, dp_dbscan_labels, f"DPC DBSCAN (epsilon={EPSILON})", dims=DIMS)
dpc.data.plot_clusters(X_low_dim, pkm_labels, f"Private KMeans (epsilon={EPSILON})", dims=DIMS)
dpc.data.plot_clusters(X_low_dim, dpdata_kmeans_labels, f"KMeans (noisy data, epsilon={EPSILON})", dims=DIMS)
dpc.data.plot_clusters(X_low_dim, dpdata_dbscan_labels, f"DBSCAN (noisy data, epsilon={EPSILON})", dims=DIMS)


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import dpclustering as dpc
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

# PARAMETERS:
EPSILON = 0.1
SEED = None
DIMS = 2

# ATTRIBUTES:
AGE = 32
SEX = 0 # male = 0, female = 1
BMI = 25.0
CHILDREN = 0
SMOKER = 0 # smoker = 1, non-smoker = 0
REGION = 0 # northeast = 0, northwest = 1, southeast = 2, southwest = 3

np.random.seed(SEED)

# LOAD DATASET

df = dpc.data.load_csv("csv/insurance.csv")

df["sex"] = df["sex"].map({"male": 0, "female": 1})
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})

region_encoded = pd.get_dummies(df["region"], prefix="region")
df = pd.concat([df.drop("region", axis=1), region_encoded], axis=1)

features = ["age", 'sex', 'bmi', 'children', 'smoker'] + list(region_encoded.columns)
X = df[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMEANS CLUSTERING
kmeans = KMeans(n_clusters=4)
kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_
df["cluster"] = kmeans.labels_

dpc_kmeans = dpc.kmeans.KMeans(X_scaled, k=4, b=3, max_iter=2000)
dpc_kmeans.fit()
dp_centroids = dpc_kmeans.add_noise_to_centroids(epsilon=EPSILON, release_labels=False)

def clip_rows(X, clip_norm):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    scaling = np.minimum(1, clip_norm / norms)
    return X * scaling
X_scaled = clip_rows(X_scaled, 3)
X_scaled_noisy = dpc.data.add_noise_to_data(X_scaled, epsilon=EPSILON, sensitivity=2 * 3, b=3)
dpdata_kmeans = KMeans(n_clusters=4)
dpdata_kmeans.fit_predict(X_scaled_noisy)
dpdata_centroids = kmeans.cluster_centers_
# PREDICTIONS

region_one_hot = [0, 0, 0, 0]
region_one_hot[REGION] = 1

new_person = [AGE, SEX, BMI, CHILDREN, SMOKER] + region_one_hot
new_person_scaled = scaler.transform([new_person])

cluster_label = kmeans.predict(new_person_scaled)[0]
cluster_label_dp = dpc_kmeans.predict(new_person_scaled)
cluster_label_dpdata = kmeans.predict(new_person_scaled)[0]

print("Original cluster label:", cluster_label)
print("DP cluster label:", cluster_label_dp)
print("DPdata cluster label:", cluster_label_dpdata)

cluster_means = df.groupby("cluster")["charges"].mean()

expected_cost = round(cluster_means[cluster_label])
expected_cost_dp = round(cluster_means[cluster_label_dp])
expected_cost_dpdata = round(cluster_means[cluster_label_dpdata])
print("Expected cost (original):", expected_cost)
print("Expected cost (DP):", expected_cost_dp)
print("Expected cost (DPdata):", expected_cost_dpdata)
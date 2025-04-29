import pandas as pd
import numpy as np

class Data:
    @staticmethod
    def load_csv(file_path):
        return pd.read_csv(file_path)

    @staticmethod
    def generate_synthetic_data(b, z, k, epsilon, sensitivity=1): 
        """
        Creates a differentially private synthetic dataset based on z.

        Parameters:
        b (float): Bounds of data points in each dimension ([-b, b]^N).
        z (numpy.ndarray): Original data points.
        epsilon (float): Differential privacy parameter.
        sensitivity (float): How much a single data point can change the output.

        Returns:
        z_tilde (numpy.ndarray): The data points with added noise.
        """

        dims = z.shape[1] 
        inc = 2 * b / k # Size of each bin

        h = np.zeros((k,) * dims)

        inds = []
        for dim in range(dims):
            coord = z[:, dim]
            ind = np.minimum(k-1, (k * (coord + b) / (2 * b)).astype(int))
            inds.append(ind)
        inds = np.array(inds)

        for point_idx in range(z.shape[0]):
            h[tuple(inds[:, point_idx])] += 1

        scale = sensitivity / epsilon
        h_noisy = h + np.random.laplace(loc=0, scale=scale, size=h.shape)
        h_noisy = np.maximum(0, h_noisy)

        z_tilde = []
        it = np.nditer(h_noisy, flags=['multi_index'])
        while not it.finished:
            count = int(round(it[0].item()))
            if count > 0:
                bin_idx = it.multi_index
                for _ in range(count):
                    point = []
                    for dim in range(dims):
                        coord = -b + bin_idx[dim] * inc + np.random.uniform(0, inc)
                        point.append(coord)
                    z_tilde.append(point)
            it.iternext()

        return np.array(z_tilde)

    @staticmethod
    def add_noise_to_data(data, epsilon, sensitivity=1, noise_type='laplace', b=None):
        """
        Adds noise to the data for differential privacy.

        Parameters:
        data (numpy.ndarray): The original data to which noise will be added.
        epsilon (float): The privacy budget for differential privacy.
        noise_type (str): The type of noise to add ('laplace' or 'gaussian').
        b (float): The bounds for clipping the data points ([-b, b]^N).

        Returns:
        numpy.ndarray: The data with added noise.
        """
        if noise_type == 'laplace':
            noise = np.random.laplace(0, sensitivity/epsilon, data.shape)
        elif noise_type == 'gaussian':
            noise = np.random.normal(0, sensitivity/epsilon, data.shape)
        else:
            raise ValueError("Unsupported noise type. Use 'laplace' or 'gaussian'.")
        
        if b is not None:
            data = np.clip(data, -b, b)

        return data + noise

    @staticmethod
    def generate_clusters(
        n_clusters=3, 
        points_per_cluster=100, 
        dim=2, 
        cluster_std=0.5, 
        separation=5,
        cluster_type='spherical'
    ):
        """
        Generates synthetic clusters of data points.

        Parameters:
        n_clusters (int): Number of clusters to generate.
        points_per_cluster (int): Number of points in each cluster.
        dim (int): Number of dimensions for each point.
        cluster_std (float): Standard deviation of the clusters.
        separation (float): Minimum distance between cluster centers.
        cluster_type (str): Type of cluster ('spherical', 'elliptical', 'bridged', 'isolated').

        Returns:
        numpy.ndarray: The generated data points.
        """
        X = []
        for i in range(n_clusters):
            center = np.random.uniform(-separation, separation, size=(dim,))

            if cluster_type == 'spherical':
                cluster_points = np.random.randn(points_per_cluster, dim) * cluster_std + center
            elif cluster_type == 'elliptical':
                cov = np.diag(np.random.uniform(0.5, 2.0, size=(dim,)))
                cluster_points = np.random.multivariate_normal(center, cov, size=points_per_cluster)
            elif cluster_type == 'bridged':
                if i == 0:
                    center_start = center
                    cluster_points = np.random.randn(points_per_cluster, dim) * cluster_std + center_start
                else:
                    center_end = center
                    bridge_points = np.linspace(center_start, center_end, points_per_cluster)
                    noise = np.random.randn(points_per_cluster, dim) * (cluster_std * 0.3)
                    cluster_points = bridge_points + noise
                    center_start = center_end
            elif cluster_type == 'isolated':
                cluster_points = np.random.randn(points_per_cluster, dim) * (cluster_std * 0.2) + center
            else:
                raise ValueError("Unsupported cluster type. Use 'spherical', 'elliptical', 'bridged', or 'isolated'.")
            
            X.append(cluster_points)
        return np.vstack(X)

    
    @staticmethod
    def plot_clusters(X, labels, title="Clusters", cmap='tab10', dims=2, show=True):
        """
        Plots the clusters in either 2D or 3D.

        Parameters:
        X (numpy.ndarray): The data points to plot.
        labels (numpy.ndarray): The cluster labels for each point.
        title (str): The title of the plot.
        cmap (str): The colormap to use for the clusters.
        """
        import matplotlib.pyplot as plt

        if dims == 2:
            plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=cmap, s=15)
            plt.title(title)
            plt.xlabel("X0")
            plt.ylabel("X1")
        elif dims == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap=cmap, s=15)
            ax.set_title(title)
            ax.set_xlabel("X0")
            ax.set_ylabel("X1")
            ax.set_zlabel("X2")
        else:
            raise ValueError("Only 2D and 3D plots are supported.")
        
        if show:
            plt.show()
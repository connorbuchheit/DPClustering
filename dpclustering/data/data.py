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
    ):
        """
        Generates synthetic clusters of data points.

        Parameters:
        n_clusters (int): Number of clusters to generate.
        points_per_cluster (int): Number of points in each cluster.
        dim (int): Number of dimensions for each point.
        cluster_std (float): Standard deviation of the clusters.
        separation (float): Minimum distance between cluster centers.

        Returns:
        tuple:
            - numpy.ndarray: The generated data points.
            - numpy.ndarray: Corresponding labels for each data point.
        """
        X = []
        y = []
        for i in range(n_clusters):
            center = np.random.uniform(-separation, separation, size=(dim,))
            cluster_points = np.random.randn(points_per_cluster, dim) * cluster_std + center
            X.append(cluster_points)
            y.append(np.full(points_per_cluster, i))
        return np.vstack(X), np.concatenate(y)
    
    @staticmethod
    def plot_clusters(X, labels, title="Clusters", cmap='tab10', dims=2, show=True, centers=None, ax=None):
        """
        Plots the clusters in 2D or 3D with centers colored to match the clusters.

        Parameters:
        X (numpy.ndarray): Data points.
        labels (numpy.ndarray): Cluster labels for each point.
        title (str): Title of the plot.
        cmap (str): Colormap name.
        dims (int): 2 for 2D, 3 for 3D plot.
        show (bool): Whether to show the plot immediately.
        centers (numpy.ndarray): Coordinates of cluster centers.
        ax (matplotlib.axes.Axes): Axes to plot on. If None, a new figure is created.
        
        Returns:
        matplotlib.figure.Figure: The figure object.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        unique_labels = np.unique(labels)
        cmap_obj = plt.get_cmap(cmap)
        color_map = {label: cmap_obj(i % cmap_obj.N) for i, label in enumerate(unique_labels)}
        fig = None

        if dims == 2:
            if ax is None:
                fig, ax = plt.subplots()
            for label in unique_labels:
                mask = labels == label
                color = color_map[label]
                ax.scatter(X[mask, 0], X[mask, 1], c=[color], s=15, label=f"Cluster {label}")

                if centers is not None:
                    ax.scatter(
                        centers[label, 0], centers[label, 1],
                        marker='o', c=[color], s=120,
                        linewidths=3, edgecolors='black'
                    )

            ax.set_title(title)
            ax.set_xlabel("X0")
            ax.set_ylabel("X1")

        elif dims == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for label in unique_labels:
                mask = labels == label
                color = color_map[label]
                ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2], c=[color], s=15, label=f"Cluster {label}")

                if centers is not None:
                    ax.scatter(
                        centers[label, 0], centers[label, 1], centers[label, 2],
                        marker='o', c=[color], s=120, linewidths=3
                    )

            ax.set_title(title)
            ax.set_xlabel("X0")
            ax.set_ylabel("X1")
            ax.set_zlabel("X2")

        else:
            raise ValueError("Only 2D and 3D plots are supported.")

        if show:
            plt.show()

        return fig

    @staticmethod
    def clip_rows(X, b):
        """
        Clips the rows of the data points to be within the bounds [-b, b].

        Parameters:
        X (numpy.ndarray): The data points to clip.
        b (float): The bounds for clipping the data points.

        Returns:
        numpy.ndarray: The clipped data points.
        """
        return np.clip(X, -b, b)
    
    @staticmethod
    def quantization_loss(X, centroids):
        """
        Computes the quantization loss for a set of centroids.

        Parameters:
        X (numpy.ndarray): The original data points.
        centroids (numpy.ndarray): The centroids to evaluate.

        Returns:
        float: The quantization loss.
        """
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        min_distances = np.min(distances, axis=1)
        return np.mean(min_distances**2)
    
    @staticmethod
    def predict(X, NOI):
        """
        Predicts the cluster labels for the data points based on the neighbors
        of interest (NOI), such as cluster centers or centroids.
        
        Parameters:
        X (numpy.ndarray): The data points to predict labels for.
        NOI (numpy.ndarray): The neighbors of interest (e.g., cluster centers).

        Returns:
        numpy.ndarray: The predicted labels for each data point. 
        """
        distances = np.linalg.norm(X[:, np.newaxis] - NOI, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels
import numpy as np

class KMeans:
    def __init__(self, X, k, b, max_iter=100, tol=1e-4):
        """
        Initialize the KMeans clustering algorithm.

        Parameters:
        X (numpy.ndarray): The input X for clustering.
        k (int): The number of clusters.
        b (float): The bounds for clipping the data points ([-b, b]^N).
        max_iter (int): The maximum number of iterations.
        tol (float): The tolerance for convergence.
        """
        self.X = X
        self.k = k
        self.b = b
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

        self.n, self.d = X.shape

    def fit(self, X=None):
        """
        Perform KMeans clustering. Centroids are initialized randomly 
        within the bounds.

        Parameters:
        X (numpy.ndarray): The input X for clustering. If None, uses the 
        initialized X. The conditions for X must remain the same as in the
        constructor: All points must be in the range [-b, b]^N, X is n x d, etc.

        Returns:
        tuple: Centroids and labels for each point.
        """
        if X is None:
            X = self.X

        centroids = np.random.uniform(-self.b, self.b, (self.k, self.d))
        
        for _ in range(self.max_iter):
            # Compute distances from samples to centroids
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

            # Assign each sample to the nearest centroid
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])

            # Check for convergence
            if np.all(np.abs(new_centroids - centroids) < self.tol):
                break

            centroids = new_centroids

        self.centroids = centroids
        self.labels = labels
        return self.centroids, self.labels
    
    def balcanetal_fit(self, epsilon, T=3, X=None):
        """
        Performs KMeans clustering using the Balcan et al. method. Translated 
        code from MATLAB to Python.
        Original paper: https://proceedings.mlr.press/v70/balcan17a.html
        Original code: https://github.com/mouwenlong/dp-clustering-icml17/tree/master

        Parameters:
        epsilon (float): The privacy parameter.
        X (numpy.ndarray): The input X for clustering. If None, uses the initialized X.
        
        Returns:
        tuple: Centroids and labels for each point.
        """
        if X is None:
            X = self.X

        X = X.T

        range_ = np.sqrt(self.d) * self.b # using _ to avoid confusion with the built-in range function
        side_length = 2.01 * self.b # to ensure no points are on the boundary, we use 2.01 instead of 2
    
        mu_mean = np.mean(X, axis=1)
        for i in range(self.n):
            tmpR = np.linalg.norm(X[:, i] - mu_mean)
            if tmpR > range_:
                range_ = tmpR

        results = [{} for _ in range(T)]
        loss_iter = np.zeros(T)
        epsilon = epsilon / T
        for t in range(T):
            print(f"Round: {t}")
            p = int(np.floor(0.5 * np.log(self.n)))
            G_projection = np.random.normal(0, 1, (p, self.d)) / np.sqrt(p)
            if self.d <= 5:
                p = self.d
                G_projection = np.eye(self.d)
            y_projected = G_projection @ X
            c_candidates = self._candidate(y_projected, side_length, p, 2 * epsilon / 3)
            print(f"Candidate set finished.")
            u_centers = self._localsearch(y_projected, c_candidates, range_, p, epsilon / 12)
            print(f"Local search finished.")
            clusters = [[] for _ in range(self.k)]
            for i in range(self.n):
                minval = 1e100
                minindex = -1
                for j in range(self.k):
                    dist = np.linalg.norm(y_projected[:, i] - u_centers[:, j])
                    if dist < minval:
                        minval = dist
                        minindex = j
                clusters[minindex].append(i)

            z_centers = np.zeros((self.d, self.k))
            totalloss = 0
            for j in range(self.k):
                z_centers[:, j] = self._recover(X[:, clusters[j]], len(clusters[j]), range_, epsilon / 6)
                diffs = X[:, clusters[j]] - z_centers[:, j].reshape(-1, 1)
                totalloss += np.sum(diffs ** 2)
            print("Starting Lloyd.")
            nLloyd = 3 # For Lloyd iteration
            for lloyditer in range(nLloyd):
                clusters = [[] for _ in range(self.k)]
                for i in range(self.n):
                    minval = 1e100
                    minindex = -1
                    for j in range(self.k):
                        dist = np.linalg.norm(X[:, i] - z_centers[:, j])
                        if dist < minval:
                            minval = dist
                            minindex = j
                    clusters[minindex].append(i)

                z_centers = np.zeros((self.d, self.k))
                totalloss = 0
                for j in range(self.k):
                    z_centers[:, j] = self._recover(X[:, clusters[j]], len(clusters[j]), range_, epsilon / (2 * nLloyd))
                    diffs = X[:, clusters[j]] - z_centers[:, j].reshape(-1, 1)
                    totalloss += np.sum(diffs ** 2)

            results[t] = {
                'z_centers': z_centers,
                'clusters': clusters,
                'c_candidates': c_candidates,
                'u_centers': u_centers,
            }
            loss_iter[t] = totalloss
            print("Lloyd finished.")

        scaled_losses = -epsilon * loss_iter / 12
        max_val = np.max(scaled_losses)
        prob = np.exp(scaled_losses - max_val)  # shift to avoid underflow
        print("Probabilities:")
        print(prob)
        prob = prob / np.sum(prob)

        
        iter_selected = np.random.choice(len(prob), p=prob)

        z_centers = results[iter_selected]['z_centers']
        clusters = results[iter_selected]['clusters']
        c_candidates = results[iter_selected]['c_candidates']
        u_centers = results[iter_selected]['u_centers']
        L_loss = loss_iter[iter_selected]

        # only care about the centers
        return z_centers.T

    def _candidate(self, X, side_length, p, epsilon):
        """
        A helper function to find candidate points for clustering. Used in the 
        Balcan et al. method. Code is translated from MATLAB to Python.
        """
        T = 2
        candidates = []

        newpart = self._partition(X, side_length, p, epsilon / T)
        candidates.append(newpart)

        for t in range(T):
            print(f"{t + 1}-th trial for a candidate set.")
            offset = np.random.uniform(-side_length / 2, side_length / 2, size=(p, 1))
            shifted = X + offset @ np.ones((1, self.n))
            newpart = self._partition(shifted, side_length, p, epsilon / T)

            L = newpart.shape[1]
            newpart -= offset @ np.ones((1, L))
            candidates.append(newpart)

        return np.hstack(candidates)
    
    def _partition(self, X, side_length, p, epsilon):
        """
        A helper function to partition the data points. Used in the Balcan et 
        al. method. Code is translated from MATLAB to Python.
        """ 
        gridpoints = []
        depth = 1
        epss = epsilon / (2 * np.log(self.n))
        gamma = 2 * np.log(self.n) / epss

        thegrid = [{
            'coordinate': np.zeros((p, 1)),
            'points': list(range(self.n))
        }]

        while depth < np.log(self.n) and len(thegrid) > 0:
            L = len(thegrid)
            side_length /= 2
            newgrid = []

            for j in range(L):
                gridpoints.append(thegrid[j]['coordinate'])

                point_indices = thegrid[j]['points']
                npoints = len(point_indices)

                directions_numeric = np.sign(X[:, point_indices] - thegrid[j]['coordinate'] @ np.ones((1, npoints)))
                directions = ["".join(str(int(v)) for v in dir_vec) for dir_vec in directions_numeric.T]
                numericmap = {key: directions_numeric[:, i] for i, key in enumerate(directions)}
                cubemap = {key: [] for key in directions}

                for i, key in enumerate(directions):
                    cubemap[key].append(i)

                keyset = list(cubemap.keys())
                valset = list(cubemap.values())
                for i in range(len(cubemap)):
                    activesize = len(valset[i])

                    if activesize > gamma:
                        prob = 1 - 0.5 * np.exp(-epss * (activesize - gamma))
                    else:
                        prob = 0.5 * np.exp(-epss * (activesize - gamma))

                    prob = np.clip(prob, 0.0, 1.0)

                    if np.random.binomial(1, prob) > 0:
                        direction = numericmap[keyset[i]]
                        coordinate_shift = (side_length / 2) * direction.reshape(-1, 1)
                        tempobj = {
                            'coordinate': thegrid[j]['coordinate'] + coordinate_shift,
                            'points': [point_indices[idx] for idx in valset[i]]
                        }
                        newgrid.append(tempobj)

            thegrid = newgrid
            depth += 1

        return np.hstack(gridpoints)
    
    def _localsearch(self, X, candidate, range_, p, epsilon):
        """
        A helper function to perform local search for clustering. Used in the 
        Balcan et al. method. Code is translated from MATLAB to Python.
        """
        _, m = candidate.shape
        weightmat = np.zeros((m, self.n))
        Lambda = 2 * range_
        
        for j in range(m):
            tmp = candidate[:, j].reshape(-1, 1) - X
            weightmat[j, :] = np.sum(tmp ** 2, axis=0)

        centerid = np.random.randint(0, m, size=(self.k,))
        T = min(self.k, 20)
        recordid = np.zeros((self.k, T), dtype=int)
        recordloss = np.zeros(T)
        loss = np.sum(np.min(weightmat[centerid, :], axis=0))
        for t in range(T):
            print(f"{t + 1}-th iteration for local search.")
            gains = np.zeros((self.k, m))
            for i in range(self.k):
                for j in range(m):
                    tmpcenterid = centerid.copy()
                    tmpcenterid[i] = j

                    newloss = np.sum(np.min(weightmat[tmpcenterid, :], axis=0))
                    gains[i, j] = newloss - loss

            raw_exp = np.exp(-epsilon * gains / (Lambda ** 2 * (T + 1)))

            i, j = self._sample_discrete(raw_exp)

            centerid[i] = j

            recordloss[t] = np.sum(np.min(weightmat[centerid, :], axis=0))
            loss = recordloss[t]
            recordid[:, t] = centerid

        final_probs = np.exp(-epsilon * recordloss / (Lambda ** 2 * (T + 1)))
        final_probs /= np.sum(final_probs)
        final_iter = np.random.choice(T, p=final_probs)

        centerid = recordid[:, final_iter]
        centers = candidate[:, centerid]
        return centers
    
    def _sample_discrete(self, A):
        """
        A helper function to sample from a matrix proportional to the entry 
        value, return row and column indices. Used in the Balcan et al. method. 
        Code is translated from MATLAB to Python.
        """
        k, m = A.shape
        total_sum = np.sum(A)
        random_coin = np.random.uniform(0, total_sum)

        curr_sum = 0.0
        for i in range(k):
            for j in range(m):
                curr_sum += A[i, j]
                if curr_sum >= random_coin:
                    return i, j
                
    def _recover(self, X, n, range_, epsilon):
        """
        A helper function to recover the centers privately from a given cluster. 
        Used in the Balcan et al. method. Code is translated from MATLAB to 
        Python.
        """
        if X.size == 0:
            z = np.random.uniform(-range_, range_, size=(self.d, 1))
        else:
            z = np.sum(X, axis=1, keepdims=True) / n
            noise_scale = range_ / (epsilon * n)
            signs = 2 * np.random.binomial(1, 0.5, size=(self.d, 1)) - 1
            noise = np.random.exponential(scale=noise_scale, size=(self.d, 1)) * signs
            z += noise
        return z.flatten()

    def predict(self, x):
        """
        Given a new sample x, predict the cluster it belongs to.
        
        Parameters:
        x (numpy.ndarray): The input sample to predict the cluster for.
        
        Returns:
        int: The predicted cluster label for the input sample.
        """
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet. Please call fit() before predict().")

        distances = np.linalg.norm(x - self.centroids, axis=1)
        return np.argmin(distances)

    def kaplan_stemmer_fit(self, epsilon, delta=1e-5, beta=0.01):
        """
        Differentially Private k-means using Kaplan and Stemmer's algorithm.
        Returns final private centers.
        Paper: https://proceedings.neurips.cc/paper_files/paper/2018/file/32b991e5d77ad140559ffb95522992d0-Paper.pdf
        """
        X = self.X
        unassigned = np.ones(self.n, dtype=bool)
        all_candidates = []
        eps_per_iter = epsilon / (2 * np.log2(np.log2(self.n)))

        for i in range(int(np.log2(np.log2(self.n))) + 1):
            S_i = X[unassigned]
            if len(S_i) == 0:
                break

            centers_i = self._private_centers(S_i, eps_per_iter)
            all_candidates.extend(centers_i)

            # Assign any points close to new centers (Euclidean distance threshold)
            dists = np.min([np.linalg.norm(X - c, axis=1) for c in centers_i], axis=0)
            radius = 2 * self.b / (2 ** i)  # shrinking threshold
            newly_covered = dists < radius
            unassigned &= ~newly_covered

        all_candidates = np.array(all_candidates)
        self.centroids = self._select_k_private(all_candidates, epsilon / 2)
        return self.centroids

    def _private_centers(self, X, epsilon):
        """
        Approximate private centers via LSH binning and noisy averaging.
        """
        n, d = X.shape
        num_bins = int(np.sqrt(n))
        random_vecs = np.random.randn(num_bins, d)
        bin_ids = (X @ random_vecs.T) > 0  # shape: (n, num_bins)
        bin_hashes = bin_ids.dot(1 << np.arange(num_bins))  # hash to int

        bins = {}
        for idx, h in enumerate(bin_hashes):
            if h not in bins:
                bins[h] = []
            bins[h].append(X[idx])

        centers = []
        for pts in bins.values():
            pts = np.array(pts)
            if len(pts) < 2:
                continue
            avg = pts.mean(axis=0)
            noise = np.random.laplace(0, self.b / (epsilon * len(pts)), size=self.d)
            centers.append(avg + noise)

        return centers

    def _select_k_private(self, candidates, epsilon):
        """
        Private selection of k centers using the exponential mechanism.
        """
        scores = []
        for _ in range(len(candidates)):
            subset = candidates[np.random.choice(len(candidates), self.k, replace=False)]
            dist_sum = np.sum([np.min(np.linalg.norm(self.X - c, axis=1)) ** 2 for c in subset])
            scores.append(-dist_sum)

        scores = np.array(scores)
        scores -= np.max(scores)
        exp_scores = np.exp(epsilon * scores / (2 * self.b**2))
        probs = exp_scores / np.sum(exp_scores)

        best_idx = np.random.choice(len(scores), p=probs)
        chosen = candidates[np.random.choice(len(candidates), self.k, replace=False)]
        return chosen
import numpy as np
from numpy.typing import NDArray

class MyKMeans:
    def __init__(self, init='k-means++', n_clusters=8, random_state=None, max_iter=300, tol=0.0001):
        self.init: str = init
        self.n_clusters: int = n_clusters
        self.max_iter = max_iter
        self.random_gen: np.random.Generator = np.random.default_rng(random_state)
        self.tol = tol

    def fit(self, X) -> "MyKMeans": #лучше импортировать Self из typing на python 3.11+
        X: NDArray[np.floating] = np.asarray(X, dtype=float)

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        if self.init == 'k-means++':
            init_centroids = self.init_kmeans_plus_plus_(X)
        elif self.init == 'random':
            init_centroids = self.init_random_(X)
        else:
            raise ValueError('Unsupported init method')
        
        
        self.cluster_centers_ = self.optymize_centroids_(X, init_centroids)

        return self
    
    def predict(self, X: NDArray) -> NDArray:
        X: NDArray[np.floating] = np.asarray(X, dtype=float)

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        vectors: NDArray[np.floating] = X[:, None, :] - self.cluster_centers_
        dists: NDArray = np.linalg.norm(vectors, axis=2)
        points_belonging: NDArray = np.argmin(dists, axis=1)

        return points_belonging

    def init_random_(self, X: NDArray) -> NDArray:
        n: int = X.shape[0]
        centroid_indxs: np.array[int] = self.random_gen.choice(n, self.n_clusters, replace=False)
        return X[centroid_indxs]
    
    def init_kmeans_plus_plus_(self, X: NDArray, eps: float = 1e-12) -> NDArray:
        n: int = X.shape[0]
        centroids: list[NDArray[np.floating]] = []

        first_centroid_index: int = self.random_gen.integers(0, n)
        first_centroid: NDArray[np.floating] = X[first_centroid_index]
        
        centroids.append(first_centroid)

        min_dists: NDArray[np.floating] = None

        for i in range(self.n_clusters - 1):
            vectors: NDArray[np.floating] = X - centroids[i]

            squared_dists = np.sum(vectors ** 2, axis=1)
            
            if min_dists is not None:
                min_dists = np.minimum(min_dists, squared_dists)
            else:
                min_dists = squared_dists
                min_dists[first_centroid_index] = 0
  
            sum_of_dists: float = np.sum(min_dists)
            
            if not np.isfinite(sum_of_dists) or sum_of_dists <= eps:
                raise ValueError("Degenerate k-means++ step: sum of squared distances is ~ 0. Try to decrease number of centroids or other init method")

            probs: NDArray[np.floating] = min_dists / sum_of_dists
            probs = probs / probs.sum()

            next_centroid_index: int = self.random_gen.choice(n, p=probs, replace=False)
            next_centroid: NDArray[np.floating] = X[next_centroid_index]
            
            min_dists[next_centroid_index] = 0
            centroids.append(next_centroid)

        return np.vstack(centroids)
        

    def optymize_centroids_(self, X: NDArray, init_centroids: NDArray) -> NDArray:

        n_clusters: int = self.n_clusters
        current_centroids: NDArray = np.copy(init_centroids)
        
        for _ in range(self.max_iter):
            new_centroids = np.zeros_like(current_centroids)

            vectors: NDArray = X[:, None, :] - current_centroids
            dists: NDArray = np.linalg.norm(vectors, axis=2)
            
            points_belonging: NDArray = np.argmin(dists, axis=1)
            clusters_indices: NDArray = [np.where(points_belonging == i)[0] for i in range(n_clusters)]

            for i, cluster_indexes in enumerate(clusters_indices):

                if len(cluster_indexes) > 0:
                    new_centroid: NDArray = np.mean(X[cluster_indexes], axis=0)

                    new_centroids[i] = new_centroid
                else:
                    new_centroids[i] = current_centroids[i]
            
            if np.allclose(current_centroids, new_centroids, atol=self.tol):
                break

            current_centroids = new_centroids

        return current_centroids

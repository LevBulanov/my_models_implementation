import numpy as np
from collections import deque
from numpy.typing import NDArray
from sklearn.neighbors import BallTree


class MyDBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric="euclidean"):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def fit_predict(self, X) -> NDArray:
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        tree = BallTree(X, metric=self.metric)

        visited = np.zeros(n, dtype=bool)
        clusters = np.full(n, -1, dtype=int)
        cluster_id = 0

        for i in range(n):
            if visited[i]:
                continue

            visited[i] = True
            neigh = self._region_query(tree, X[i])

            if len(neigh) < self.min_samples:
                continue  # пока помечаем как шум

            self._expand_cluster(tree, X, neigh, cluster_id, clusters, visited)
            cluster_id += 1

        return clusters

    def _region_query(self, tree: BallTree, point: NDArray) -> NDArray:
        return tree.query_radius(point.reshape(1, -1), r=self.eps)[0]

    def _expand_cluster(self, tree, X, neighbours, cluster_id, clusters, visited):
        queue = deque(neighbours)
        in_queue = set(neighbours)  # чтобы не добавлять дубликаты

        while queue:
            idx = queue.popleft()

            if not visited[idx]:
                visited[idx] = True
                neigh = self._region_query(tree, X[idx])

                if len(neigh) >= self.min_samples:
                    for nb in neigh:
                        # добавляем в очередь только если:
                        # - ещё не в кластере
                        # - ещё не в очереди
                        if clusters[nb] == -1 and nb not in in_queue:
                            in_queue.add(nb)
                            queue.append(nb)

            # назначаем кластер, если точка была шумом
            if clusters[idx] == -1:
                clusters[idx] = cluster_id







                

        








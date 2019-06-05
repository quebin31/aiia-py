import math
import random
import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple;

DistanceInfo = namedtuple('DistanceInfo', ['index', 'distance']);

class KMeans:
    def clusterize(centroids, X):
        clusters = []

        for x in X:
            min_dist = DistanceInfo(index=0, distance=math.inf)
            for index, centroid in enumerate(centroids):
                distance = np.linalg.norm(centroid - x)
                if distance < min_dist.distance:
                    min_dist = DistanceInfo(index, distance)

            clusters.append(min_dist.index)

        return np.array(clusters)

    def generate_random_centroids(X, k):
        selected  = []
        centroids = []
        no_points = X.shape[0]

        while len(selected) != k:
            generated = None
            while True:
                generated = random.randint(0, no_points - 1)
                if generated not in selected:
                    break

            selected.append(generated)
            centroids.append(X[generated])

        return np.array(centroids)

    def generate_fixed_centroids(X, distances, k):
        centroids = []
        for i in range(k):
            classi = X[distances == i]
            centroids.append(np.mean(classi, axis=0))

        return np.array(centroids)

    def fit(self, X, k, epochs):
        centroids = KMeans.generate_random_centroids(X, k)
        distances = KMeans.clusterize(centroids, X)

        for _ in range(epochs):
            distances = KMeans.clusterize(centroids, X)
            centroids = KMeans.generate_fixed_centroids(X, distances, k)

        return centroids


if __name__ == '__main__':
    from sklearn.datasets.samples_generator import make_blobs
    centers = [[0,0], [5,5], [10,10]]
    X, _ = make_blobs(n_samples=100, centers=centers, cluster_std=1)

    model = KMeans()
    centroids = model.fit(X, 3, 50)
    print(centroids)
    plt.scatter(X[:, 0], X[:, 1], s=150)
    plt.scatter(centroids[:, 0], centroids[:, 1], color='k', marker='*', s=150)
    plt.show()


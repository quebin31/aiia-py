import numpy as np
import matplotlib.pyplot as plt

class MeanShift:
    def __init__(self, radius=4):
        self.radius = radius

    def shift(self, centroids, X):
        new_centroids = []
        for index, centroid in enumerate(centroids):
            inside_radius = []
            for x in X:
                if np.linalg.norm(x - centroid) < self.radius:
                    inside_radius.append(x)

            new_centroid = np.mean(inside_radius, axis=0)
            new_centroids.append(new_centroid)

        new_centroids = np.unique(new_centroids, axis=0)
        return new_centroids
    
    def done(self, old_centroids, new_centroids):
        equal_length = len(old_centroids) == len(new_centroids)
        return equal_length and np.array_equal(old_centroids, new_centroids)

    def fit(self, X):
        centroids = X
        epoch = 1
        while True:
            old_centroids = centroids
            centroids     = self.shift(centroids, X)
            if self.done(old_centroids, centroids):
                break

            epoch += 1 

        return centroids


if __name__  == '__main__':
    from sklearn.datasets.samples_generator import make_blobs
    centers = [[0,0], [5,5], [10,10]]
    X, _ = make_blobs(n_samples=100, centers=centers, cluster_std=1)

    model = MeanShift(radius=3.5)
    centroids = model.fit(X)
    print(centroids)
    
    plt.scatter(X[:, 0], X[:, 1], s=100)
    plt.scatter(centroids[:, 0], centroids[:, 1], color='k', marker='*', s=100)
    plt.show()

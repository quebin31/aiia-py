import numpy as np

from scipy.stats import multivariate_normal as mv_normal

class GMM:
    def __init__(self, no_clusters):
        self.no_clusters = no_clusters
        self.mu = None
        self.cv = None 
        self.ak = None
    
    def init_params(self, X, rand_range=(0,1)):
        d = X.shape[1]
        self.initial_means = np.random.uniform(*rand_range, (self.no_clusters, d))
        self.initial_covs  = np.random.uniform(*rand_range, (self.no_clusters, d, d))
        self.initial_aks   = np.random.uniform(*rand_range, (self.no_clusters,))

        return (self.initial_means, self.initial_covs, self.initial_aks)
      
    def expectation_step(self, X):
        m = X.shape[0]
        self.probs = np.zeros((m, self.no_clusters))

        for c in range(self.no_clusters):
            self.probs[:,c] = self.ak[c] * mv_normal.pdf(X, self.mu[c,:], self.cv[c])

        probs_norm = np.sum(self.probs, axis=1)[:, np.newaxis]
        self.probs /= probs_norm

    def maximization_step(self, X):
        self.ak = np.mean(self.probs, axis=0)
        self.mu = (self.probs.T @ X) / np.sum(self.probs, axis=0)[:, np.newaxis]

        for c in range(self.no_clusters):
            x = X - self.mu[c, :]
            probs_diag = np.diag(self.probs[:,c])

            cv = x.T @ probs_diag @ x
            self.cv[c, :, :] = cv / np.sum(self.probs, axis=0)[:, np.newaxis][c]
        
    def compute_loss(self, X):
        m = X.shape[0]
        self.loss = np.zeros((m, self.no_clusters))

        for c in range(self.no_clusters):
            dist = mv_normal(self.mu[c], self.cv[c], allow_singular=True)
            self.loss[:, c] = self.probs[:, c] * (np.log(self.ak[c] + 1e-8) +
                                                  dist.logpdf(X) -
                                                  np.log(self.probs[:, c] + 1e-8))
            self.loss = np.sum(self.loss)
      
    def fit(self, X, epochs=200, print_each=10, rand_range=(0,1)): 
        self.mu, self.cv, self.ak = self.init_params(X, rand_range)

        for epoch in range(epochs):
            self.expectation_step(X)
            self.maximization_step(X)
            self.compute_loss(X)

            if epoch % epochs == 0:
                print(f'Epoch {epoch}, loss = {self.loss}')

        return self.loss

    def predict(self, X):
        labels = np.zeros((X.shape[0], self.no_clusters))

        for c in range(self.no_clusters):
            labels[:, c] = self.ak[c] * mv_normal.pdf(X, self.mu[c, :], self.cv[c])

        labels = labels.argmax(1)
        return labels

if __name__ == '__main__':
    from sklearn.datasets.samples_generator import make_classification
    X, _ = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1, n_classes=3)


    model = GMM(3)
    model.fit(X)
    model.predict(X)

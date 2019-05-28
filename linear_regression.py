import numpy as np

class LinearRegression:
    def __init__(self):
        self.weigths = np.array([])

    def predict(self, X):
        return X @ self.weigths

    def error(self, X, y):
        prediction = self.predict(X)
        distance = prediction - y
        return (distance @ distance) / (2 * X.shape[0])

    def update_weigths(self, X, y, alpha):
        prediction = self.predict(X)
        derivative = (prediction - y) @ X
        self.weigths = self.weigths - alpha/X.shape[0] * derivative

    def fit(self, X, y, alpha, tolerance, range_gen=(0, 1), print_at=50):
        initial = self.weigths = np.random.uniform(range_gen[0], range_gen[1], (X.shape[1],))
        error = [self.error(X,y), 0.0]
        self.update_weigths(X, y, alpha)
        error[1] = self.error(X, y)

        epoch = 1
        while abs(error[1] - error[0]) >= tolerance:
            self.update_weigths(X, y, alpha)
            error[0] = error[1]
            error[1] = self.error(X, y)
            if not (epoch % print_at):
                print(f'Current error: {error[1]}')
            epoch += 1

        return (initial, self.weigths)


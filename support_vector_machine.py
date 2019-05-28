import numpy as np

class SVM:
    def __init__(self):
        self.weigths = np.array([])

    def predict(self, X):
        return X @ self.weigths

    def error(self, X, y, c, penalty):
        regularization = penalty * (self.weigths @ self.weigths)
        hinge_term = 1 - y * self.predict(X)
        hinge_loss = np.maximum(np.zeros(hinge_term.shape[0]), hinge_term)
        
        return c * np.mean(hinge_loss) + regularization

    def update_weigths(self, X, y, c, penalty, alpha):
        regularization = penalty * 2 * self.weigths
        subderivative = -y.reshape((-1,1)) * X
        subderivative[y * self.predict(X) >= 1] = np.zeros(X.shape[1])

        derivative = regularization + c * np.mean(subderivative, axis=0)
        self.weigths = self.weigths - alpha * derivative


    def fit(self, X, y, c, penalty, alpha, tolerance, range_gen=(0, 1), print_at=50):
        self.weigths = np.random.uniform(range_gen[0], range_gen[1], (X.shape[1],))
        initial_weigths = self.weigths

        error = [0, 0]
        error[0] = self.error(X, y, c, penalty)
        self.update_weigths(X, y, c, penalty, alpha)
        error[1] = self.error(X, y, c, penalty)

        epoch = 1
        while abs(error[1] - error[0]) >= tolerance:
            self.update_weigths(X, y, c, penalty, alpha)
            error[0] = error[1]
            error[1] = self.error(X, y, c, penalty)
            penalty *= 1 / epoch
            # alpha *= 1 / epoch
            if not (epoch % print_at):
                print(f'Current error: {error[1]}')
            epoch += 1

        return (initial_weigths, self.weigths)


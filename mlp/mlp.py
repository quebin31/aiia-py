import numpy as np
from layer import linear, relu, sigmoid, Layer

class MLP:
    def __init__(self, layers):
        if len(layers) < 2:
            raise RuntimeError('At least two layers (input and output)')

        if layers[0].activation != linear:
            raise RuntimeError('Input layer cannot have an activation (leave it linear)')


        self.layers = layers

    def _init_weigths(self, interval):
        for i in range(1, len(self.layers)):
            self.layers[i].weights = np.random.uniform(
                # Interval (low, high)
                interval[0],
                interval[1],

                # Weight matrix, one extra column for bias
                (self.layers[i].size, self.layers[i - 1].size + 1))

    def fordward(self, x):
        self.layers[0].units = x
        self.layers[0].extend() # Add bias unit

        for i, _ in enumerate(self.layers[1:]):
            self.layers[i].units = self.layers[i].weights @ self.layers[i - 1].units
            self.layers[i].activate()

            # Do not extend last layer
            if i != len(self.layers):
                self.layers[i].extend()


        return self.layers[-1].units


    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self.fordward(x))

        return np.array(predictions)

    def error(self, X, Y, cost_type):
        if cost_type == 'mse':
            predictions  = self.predict(X)
            mean_sqerror = np.sum((predictions - Y) ** 2, axis=1)
            return 0.5 * np.mean(mean_sqerror)

        if cost_type == 'cross_entropy':
            predictions = self.predict(X)
            class1  = Y * np.log(predictions + 1.0e-10) 
            class0 = (1 - Y) * np.log(1 - predictions + 1.0e-10)
            cross_cost = np.sum(class1 + class0, axis=1)
            return -np.mean(cross_cost)

        raise RuntimeError(f'Unknown cost funcion {cost_type}')

    def backprop(self, x, y, cost_type):
        prediction = self.fordward(x)

        grad = {
            'mse': prediction - y,
            'cross_entropy': (prediction - y) / prediction * (1 - prediction)
        }


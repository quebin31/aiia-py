import numpy as np

# Activation functions
relu    = lambda x: np.maximum(np.zeros(x.size), x)
linear  = lambda x: x
sigmoid = lambda x: 1 / (1 + np.exp(-x))

# Exceptions 
class IllformedArchitecture(BaseException):
    pass

class DifferentActivation(BaseException):
    pass

class NotRecognizedActivation(BaseException):
    pass

# Layer abstraction
class Layer:
    def __init__(self, no_neurons, activation):
        if activation != 'relu' or activation != 'linear' or activation != 'sigmoid':
            raise NotRecognizedActivation

        self.size = no_neurons
        self.activation = activation


# Neural network abstraction
class NeuralNetwork:
    def __init__(self, layers):
        if len(layers) == 0:
            raise IllformedArchitecture

        if len(layers) < 2:
            raise IllformedArchitecture

        layer_activation = layers[0].activation

        for layer in layers:
            if layer.activation != layer_activation:
                raise DifferentActivation

        self.layers = layers
        self.weigths = None
        self.activation = layer_activation

        self.activationfn = None
        if layer_activation == 'relu':
            self.activationfn = relu
        elif layer_activation == 'linear':
            self.activationfn = linear
        else:
            self.activationfn = sigmoid

    def _init_weigths(self, range_gen):
        for i in range(0, len(self.layers) - 1):
            matrix = np.random.uniform(
                range_gen[0], 
                range_gen[1],
                (self.layers[i].size, self.layers[i + 1].size))

            self.weigths.append(matrix)

        return self.weigths

    def forward(self, X):
        result = X
        for i in range(0, len(self.layers) - 1):
            result = self.activationfn(result @ self.weigths[i])

        return result

    def predict(self, X):
        return self.forward(X)

    def error(self, X, y):
        if self.activation == 'relu':
            raise NotImplementedError

        if self.activation == 'linear':
            prediction = self.predict(X)
            distance = prediction - y
            mse = distance * distance
            return 0.5 * np.sum(np.sum(mse, axis=1)) * X.shape[0]

        if self.activation == 'sigmoid':
            prediction = self.predict(X)
            left_part = np.sum(np.sum(y * np.log(prediction), axis=1))
            rigth_part = np.sum(np.sum((1 - y) * np.log(1 - prediction), axis=1))
            return -(left_part + rigth_part) / X.shape[0]

    def backprop(self):
        # going to return a list with all the gradients
        pass

    def update_weigths(self, X, y, alpha):
        gradients = self.backprop()
        for (i, weigth) in enumerate(self.weigths):
            self.weigths[i] = weigth - alpha * gradients[i]

    def fit(self, X, y, alpha, tolerance, range_gen=(0, 1), print_each=50):
        initial_weigths =  self._init_weigths(range_gen)
        error = []
        error.append(self.error(X, y))
        self.update_weigths(X, y, alpha)
        error.append(self.error(X, y))

        epoch = 1
        while abs(error[1] - error[0]) >= tolerance:
            self.update_weigths(X, y, alpha)
            error[0] = error[1]
            error[1] = self.error(X, y)
            if not (epoch % print_each):
                print(f'Current error: {error[1]}')
            epoch += 1 

        return initial_weigths

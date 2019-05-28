import numpy as np

# Activation functions
relu    = lambda x: np.maximum(np.zeros(x.size), x)
linear  = lambda x: x
sigmoid = lambda x: 1 / (1 + np.exp(-x))

# Exceptions 
class IllformedArchitectureError(BaseException):
    pass

class DifferentActivationError(BaseException):
    pass

class NotRecognizedActivationError(BaseException):
    pass

class InputLayerActivationError(BaseException):
    pass

# Layer abstraction
class Layer:
    def __init__(self, no_neurons, activation):
        if activation != 'relu' and activation != 'linear' and activation != 'sigmoid':
            raise NotRecognizedActivationError

        self.size = no_neurons
        self.value = []
        self.activation = activation


# Neural network abstraction
class NeuralNetwork:
    def __init__(self, layers):
        if len(layers) == 0:
            raise IllformedArchitectureError

        if len(layers) < 2:
            raise IllformedArchitectureError

        if layers[0].activation != 'linear':
            raise InputLayerActivationError

        layer_activation = layers[1].activation

        for i in range(1, len(layers)):
            if layers[i].activation != layer_activation:
                raise DifferentActivationError

        self.layers = layers
        self.weigths = []
        self.activation = layer_activation

        if layer_activation == 'relu':
            raise NotImplementedError
            self.activationfn = relu
        elif layer_activation == 'linear':
            self.activationfn = linear
        else:
            self.activationfn = sigmoid

    def init_weigths(self, range_gen):
        self.weigths = []
        for i in range(0, len(self.layers) - 1):
            matrix = np.random.uniform(
                range_gen[0], 
                range_gen[1],
                (self.layers[i].size, self.layers[i + 1].size))

            self.weigths.append(matrix)

        return self.weigths

    def forward(self, x):
        self.layers[0].value = x
        for i in range(1, len(self.layers)):
            self.layers[i].value = self.activationfn(self.layers[i - 1].value @ self.weigths[i - 1])
        return self.layers[-1].value

    def predict(self, X):
        return np.array([self.forward(x) for x in X])

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

    def backprop(self, X, y):
        if self.activation == 'relu':
            raise NotImplementedError
        if self.activation == 'linear':
            raise NotImplementedError

        if self.activation == 'sigmoid':
            # m is number of examples
            # X in R^{m·n}
            # y in R^{m·o} 
            prediction = self.predict(X) # in R^{m·o}
            gradient = np.sum((prediction - y) / prediction * (prediction - 1), axis=0) # in R^o
            for i in range(len(self.layers) - 1, 0, -1):
                gradient = gradient 

            return gradient

    def update_weigths(self, X, y, alpha):
        gradients = self.backprop()
        for (i, weigth) in enumerate(self.weigths):
            self.weigths[i] = weigth - alpha * gradients[i]

    def fit(self, X, y, alpha, tolerance, range_gen=(0, 1), print_each=50):
        initial_weigths =  self.init_weigths(range_gen)
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

if __name__ == '__main__':
    model = NeuralNetwork([
        Layer(3, 'linear'),
        Layer(2, 'sigmoid'),
        Layer(2, 'sigmoid'),
    ])

    model.init_weigths((0, 1))
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    y = np.array([
        [1, 0],
        [1, 0],
        [0, 1],
    ])

    print(model.forward(X))
    print(model.layers[0].value)
    #print(model.backprop(X, y))

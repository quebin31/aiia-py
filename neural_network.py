import numpy as np

# Activation functions
relu    = lambda x: np.maximum(np.zeros(x.size), x)
linear  = lambda x: x
sigmoid = lambda x: 1 / (1 + np.exp(-x))

# Exceptions 
class IllformedArchitectureError(Exception):
    pass

class DifferentActivationError(Exception):
    pass

class NotRecognizedCostError(Exception):
    pass

class NotRecognizedActivationError(Exception):
    pass

class InputLayerActivationError(Exception):
    pass

# Layer abstraction
class Layer:
    def __init__(self, no_neurons, activation):
        self.activation = {
            'relu': relu,
            'linear': linear,
            'sigmoid': sigmoid
        }.get(activation, None) 

        if not self.activation:
            raise NotRecognizedActivationError(f'Unknown activation function {activation}')

        self.value = []
        self.size = no_neurons

    def activate(self):
        self.value = self.activation(self.value)

# Neural network abstraction
class NeuralNetwork:
    def __init__(self, layers):
        # Check that there are at least two layers
        if len(layers) < 2:
            raise IllformedArchitectureError

        # Check that the first layer (input layer) has a linear activation
        if layers[0].activation != linear:
            raise InputLayerActivationError(f'First layer needs to have linear activation')

        # Initialize member variables
        self.layers = layers
        self.weigths = []


    # Randomly initialize weigths from the specified range_gen
    def init_weigths(self, range_gen):
        self.weigths = []
        for i in range(0, len(self.layers) - 1):
            matrix = np.random.uniform(
                range_gen[0], 
                range_gen[1],
                (self.layers[i].size, self.layers[i + 1].size))

            self.weigths.append(matrix)

        return self.weigths

    # Forward could be applied to feature matrix or a single example
    # Case X is feature matrix:
    #   It'll calculate forward propagation for every example and will set
    #   self.layers[i].value to be a matrix containing h^(j) in each row for each 
    #   example X[j], e.g.
    #   
    #       self.layers[i].value = [h^(i)_0, ..., h^(i)_j]
    #           where h^(i) in R^{self.layers[i].size}
    #           where j is the number of examples
    #
    # Case X is a single example:
    #   It'll calculate forward propagation for this example and will set
    #   self.layers[i].value to be a vector containing h^(i) for this example
    def forward(self, X):
        self.layers[0].value = X
        for i in range(1, len(self.layers)):
            self.layers[i].value = self.layers[i - 1].value @ self.weigths[i - 1]
            self.layers[i].activate()

        return self.layers[-1].value

    # Just the same as self.forward
    def predict(self, X):
        return self.forward(X)

    # Calculate error based on the default activation for this neural network
    # TODO: This can be changed to accept 'mse', 'cross_entropy', etc.
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

    # Calculate the gradient using the backpropagation algorithm
    # TODO: This can be changed to depend on the error function type ('mse',
    # 'cross_entropy', etc) and working with layers with different activation functions
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

    # Update weigths using gradient descent
    # TODO: This can be changed to stochastic gradient descent
    def update_weigths(self, X, y, alpha):
        gradients = self.backprop()
        for (i, weigth) in enumerate(self.weigths):
            self.weigths[i] = weigth - alpha * gradients[i]

    # Main function, make the neural network 'fit' the example data to the desired data
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

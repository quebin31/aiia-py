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
            #'relu': relu,
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
            raise IllformedArchitectureError(f'We need at least two layers')

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

    # Calculate error based on the error_type (values: 'mse', 'cross_entropy')
    def error(self, X, y, error_type):
        if error_type == 'mse':
            prediction = self.predict(X)
            distance = prediction - y
            mse = distance * distance
            return 0.5 * np.sum(np.sum(mse, axis=1)) * X.shape[0]

        if error_type == 'cross_entropy':
            prediction = self.predict(X)
            left_part = np.sum(np.sum(y * np.log(prediction), axis=1))
            rigth_part = np.sum(np.sum((1 - y) * np.log(1 - prediction), axis=1))
            return -(left_part + rigth_part) / X.shape[0]

        raise NotRecognizedActivationError(f'Unknown cost function {error_type}')

    # Calculate the gradient using the backpropagation algorithm 
    def backprop(self, X, y, error_type):
        # m is number of examples
        # X in R^{m·n}
        # y in R^{m·o} 
        prediction = self.predict(X)

        # Calculate cost gradient w.r.t. output 
        gradient = {
            'mse': np.sum(prediction - y, axis=0), # in R^o
            'cross_entropy': np.sum((prediction - y) / prediction * (prediction - 1), axis=0), # in R^o
        }[error_type]

        gradients = []

        # Propagate the gradient
        for l in range(len(self.layers) - 1, 0, -1):
            # Not expecting any other activation
            activation_derivative = {
                #relu: TODO, 
                linear: 1,
                sigmoid: self.layers[l].value * (self.layers[l].value - 1)
            }[self.layers[l].activation]

            gradient = gradient * np.sum(activation_derivative, axis=0)
            gradient = gradient.reshape((-1, 1)) @ np.sum(self.layers[l - 1].value,
                                                          axis=0).reshape((1, -1))
            gradients.append(gradient)
            #gradient = gradient @ self.weigths[l - 1]

        return gradients

    # Update weigths using gradient descent
    # TODO: This can be changed to stochastic gradient descent
    def update_weigths(self, X, y, alpha, error_type):
        gradients = self.backprop(X, y, error_type)
        for (i, weigth) in enumerate(self.weigths):
            self.weigths[i] = weigth - alpha * gradients[i]

    # Main function, make the neural network 'fit' the example data to the desired data
    def fit(self, X, y, alpha, tolerance, error_type, range_gen=(0, 1), print_each=50):
        initial_weigths =  self.init_weigths(range_gen)
        error = []
        error.append(self.error(X, y, error_type))
        self.update_weigths(X, y, alpha, error_type)
        error.append(self.error(X, y, error_type))

        epoch = 1
        while abs(error[1] - error[0]) >= tolerance:
            self.update_weigths(X, y, alpha, error_type)
            error[0] = error[1]
            error[1] = self.error(X, y, error_type)
            if not (epoch % print_each):
                print(f'Current error: {error[1]}')
            epoch += 1 

        return initial_weigths

if __name__ == '__main__':
    model = NeuralNetwork([
        Layer(3, 'linear'),
        Layer(2, 'sigmoid'),
        Layer(3, 'sigmoid'),
    ])

    model.init_weigths((0, 1))
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    y = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    print('Forward propagation')
    print(model.forward(X))

    print('Model\'s layers')
    for layer in model.layers:
        print(layer.value)

    print('Error function')
    print(model.error(X, y, 'cross_entropy'))

    model.fit(X, y, 0.001, 0.00001, 'cross_entropy')

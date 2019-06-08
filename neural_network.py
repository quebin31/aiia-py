import numpy as np

# Activation functions
linear  = lambda x: x
sigmoid = lambda x: 1 / (1 + np.exp(-x))

DEBUG = True

def debug(x):
    if DEBUG:
        print(x)
        input("Press enter")

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

# Layer abstraction class
class Layer:
    def __init__(self, size, activation):
        # Check activation
        self.activation = {
            'linear'  : linear,
            'sigmoid' : sigmoid
        }.get(activation, None) 

        if not self.activation:
            raise NotRecognizedActivationError(f'Unknown activation function {activation}')

        self.units   = []
        self.weigths = []
        self.size    = size

    def activate(self):
        # Activate units, except bias unit
        self.units[1:] = self.activation(self.units[1:])

    def activation_derivative(self):
        derivative = {
            linear: 1.,
            sigmoid: self.units[1:] * (1 - self.units[1:])
        }[self.activation]

        return derivative

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
        self.layers  = layers

    # Randomly initialize weigths from the specified range_gen
    def init_weigths(self, range_gen):
        for i in range(1, len(self.layers)):
            self.layers[i].weigths = np.random.uniform(
                range_gen[0],
                range_gen[1],
                (self.layers[i].size, self.layers[i - 1].size + 1))

    # Apply forward propagation for an example
    def forward_propagation(self, x):
        self.layers[0].units = np.concatenate(([1.], x))

        for i in range(1, len(self.layers)):
            self.layers[i].units = np.concatenate(
                ([1.], self.layers[i].weigths @ self.layers[i - 1].units))
            self.layers[i].activate()

        # Last layer doesn't need a bias
        return self.layers[-1].units[1:]

    # Apply forward propagation for many examples
    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self.forward_propagation(x))

        return np.array(predictions)

    # Calculate error based on the error_type (values: 'mse', 'cross_entropy')
    def error(self, X, Y, error_type):
        if error_type == 'mse':
            predictions = self.predict(X)
            mean_sqerr  = np.sum((predictions - Y) ** 2)
            return mean_sqerr / (2 * X.shape[0])

        if error_type == 'cross_entropy':
            predictions = self.predict(X)
            left_part   = Y * np.log(predictions)
            rigth_part  = (1 - Y) * np.log(1 - predictions)
            cross_error = np.sum(left_part + rigth_part)
            return -(cross_error / X.shape[0])

        raise NotRecognizedActivationError(f'Unknown cost function {error_type}')

    # Calculate gradients w.r.t. to only one example
    def backprop(self, x, y, error_type):
        # Update layer's units 
        prediction = self.forward_propagation(x)

        # print(prediction, y)
        # Calculate gradient of error w.r.t. last layer units (output)
        gradient = {
            'mse': prediction - y,
            'cross_entropy': (prediction - y) / prediction * (1 - prediction)  
        }[error_type]
        
        # Store weigths derivatives
        backprop = []

        for l in range(len(self.layers) - 1, 0, -1):
            activation_derivative = self.layers[l].activation_derivative()
            gradient = gradient * activation_derivative
            dweigths = gradient.reshape((-1, 1)) @ self.layers[l - 1].units.reshape((1, -1))
            backprop.append(dweigths)

            gradient = self.layers[l].weigths[:, 1:].T @ gradient

        return np.array(list(reversed(backprop)))

    # Update weigths using gradient descent
    def update_weigths(self, X, Y, alpha, error_type):
        temp_layers = self.layers

        # Take each example and it's desired output
        gradients = []
        for i in range(1, len(self.layers)):
            gradients.append(np.zeros(self.layers[i].weigths.shape))

        gradients = np.array(gradients)

        for x, y in zip(X, Y):
            gradients += self.backprop(x, y, error_type)
            # gradients /= X.shape[0]
            # debug(gradients)

        for i in range(1, len(self.layers)):
            temp_layers[i].weigths = temp_layers[i].weigths - alpha * gradients[i - 1] 

        self.layers = temp_layers

    # Main function, make the neural network 'fit' the example data to the desired data
    def fit(self, X, y, alpha, tolerance, error_type, range_gen=(0, 1), print_each=50,
            max_epoch=None):
        self.init_weigths(range_gen)
        error = []
        error.append(self.error(X, y, error_type))
        self.update_weigths(X, y, alpha, error_type)
        error.append(self.error(X, y, error_type))
        debug(error)

        epoch = 1
        while abs(error[1] - error[0]) >= tolerance:
            self.update_weigths(X, y, alpha, error_type)
            error[0] = error[1]
            error[1] = self.error(X, y, error_type)
            if not (epoch % print_each):
                print(f'Epoch {epoch}, current error: {error[1]}')

            if epoch == max_epoch:
                break

            epoch += 1 

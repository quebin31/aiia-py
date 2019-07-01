import numpy as np

# Activation functions
relu    = lambda x: np.maximum(np.zeros(x.shape), x)
linear  = lambda x: x
sigmoid = lambda x: 1. / (1. + np.exp(-x)) 

def relu_derivative(x):
    d = np.zeros(x.shape)
    d[x != 0] = 1.
    return d

class Layer:
    def __init__(self, size, activation='linear'):
        # Check activation function       
        self.activation = {
            'relu'    : relu,
            'linear'  : linear,
            'sigmoid' : sigmoid
        }.get(activation)

        if not self.activation:
            raise RuntimeError(f'Unknown activation function {activation}')

        self.size    = size
        self.units   = None
        self.weigths = None

    def activate(self):
        self.units = self.activation(self.units)

    def extend(self):
        self.units = np.concatenate(([1.], self.units))

    def derivative(self, bias=True):
        start = 1 if bias else 0

        derivative = {
            relu: relu_derivative(self.units[start:]),
            linear: 1.,
            sigmoid: self.units[start:] * (1. - self.units[start:] + 1.0e-10)
        }.get(self.activation)

        return derivative

import numpy as np

from enum import Enum


class Activations(Enum):
    identity = 1
    sigmoid = 2
    relu = 3

    @staticmethod
    def get_activation_function(activation):
        if activation == Activations.identity:
            return Identity()
        if activation == Activations.sigmoid:
            return Sigmoid()
        if activation == Activations.relu:
            return ReLu()
        raise ValueError('no such activation')


class Activation:

    def __init__(self):
        pass

    def activate(self, x):
        raise NotImplementedError('should have implemented this')

    def prime(self, x):
        raise NotImplementedError('should have implemented this')


class Identity(Activation):

    def activate(self, x):
        return x

    def prime(self, x):
        return 1


class Sigmoid(Activation):

    def activate(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def prime(self, x):
        return self.activate(x) * (1.0 - self.activate(x))


class ReLu(Activation):

    def activate(self, x):
        return np.maximum(x, 0.0, x)

    def prime(self, x):
        return np.where(x > 0, 1.0, 0.0)

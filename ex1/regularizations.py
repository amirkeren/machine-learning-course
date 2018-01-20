import numpy as np


class Regularization:

    def __init__(self, alpha=0):
        self.alpha = alpha

    def regularize(self, lst):
        raise NotImplementedError('should have implemented this')


class NoRegularization(Regularization):

    def regularize(self, lst):
        return 0


class L1(Regularization):

    def regularize(self, lst):
        return self.alpha * np.sum([abs(x) for x in lst])


class L2(Regularization):

    def regularize(self, lst):
        def square(arr):
            return map(lambda x: x ** 2, arr)

        return self.alpha * np.sum(square(lst))

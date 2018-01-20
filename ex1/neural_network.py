import numpy as np

from activations import Activations
from regularizations import NoRegularization


class _Layer:
    # TODO - add regularization
    def __init__(self, nodes, activation_function, regularization, keep_probability):
        assert 0 <= keep_probability <= 1, 'keep probability must be between 0 and 1'
        self.nodes = nodes
        self.activation_function = Activations.get_activation_function(activation_function)
        self.regularization = regularization
        self.keep_probability = keep_probability


class NeuralNetwork:
    def __init__(self, learning_rate):
        assert learning_rate > 0, 'learning rate must be a greater than 0'
        self._layers = self._biases = self._weights = []
        self._lr = learning_rate
        self.is_initialized = False

    def add_layer(self, nodes, activation_function=Activations.identity, regularization=NoRegularization(),
                  keep_probability=1):
        assert nodes > 0, 'number of nodes in layer must be a greater than 0'
        self._layers.append(_Layer(nodes, activation_function, regularization, keep_probability))

    def train(self, features, targets):
        assert len(features) > 0 and len(targets) > 0, 'features and targets size must be greater than 0'
        assert len(features) == len(targets), 'features and targets are not the same size'
        if not self.is_initialized:
            self._initialize()
        n_records = len(features)
        biases = [np.zeros(biases.shape) for biases in self._biases]
        weights = [np.zeros(weights.shape) for weights in self._weights]
        dropout_masks = []
        for x, y in zip(features, targets):
            activations, results = self._forward_pass(x, dropout_masks)
            delta_biases, delta_weights = self._backward_propagation(activations, results, y, dropout_masks)
            biases = [new_biases + delta_new_biases for new_biases, delta_new_biases in zip(biases, delta_biases)]
            weights = [new_weights + delta_new_weights for new_weights, delta_new_weights in
                       zip(weights, delta_weights)]
        self._weights = [weights - (self._lr * (new_weights +
                                                self._layers[layer].regularization.regularize(new_weights))) / n_records
                         for layer, (weights, new_weights) in enumerate(zip(self._weights, weights))]
        self._biases = [biases - (self._lr * new_biases) / n_records for biases, new_biases in
                        zip(self._biases, biases)]

    def predict(self, x):
        if not self.is_initialized:
            self._initialize()
        for layer, (biases, weights) in enumerate(zip(self._biases, self._weights)):
            x = self._layers[layer + 1].activation_function.activate(np.dot(weights, x) + biases)
        return x

    def _initialize(self):
        if self.is_initialized:
            return
        assert len(self._layers) > 0, 'no layers added to the network'
        self._biases = [np.random.randn(layer.nodes, 1) for layer in self._layers[1:]]
        self._weights = [np.random.randn(layer2.nodes, layer1.nodes) for layer1, layer2 in
                         zip(self._layers[:-1], self._layers[1:])]
        self.is_initialized = True

    def _forward_pass(self, x, dropout_masks):
        create_dropout_masks = False
        if len(dropout_masks) == 0:
            create_dropout_masks = True
        activations = [self._layers[0].activation_function.activate(x)]
        results = []
        for layer, (biases, weights) in enumerate(zip(self._biases, self._weights)):
            result = np.dot(weights, activations[layer]) + biases
            if create_dropout_masks:
                mask = self._dropout(activations[layer], self._layers[layer].keep_probability)
                dropout_masks.append(mask)
            else:
                mask = dropout_masks[layer]
            activations[layer] *= mask
            results.append(result)
            activations.append(self._layers[layer + 1].activation_function.activate(result))
        return activations, results

    def _backward_propagation(self, activations, results, y, dropout_masks):
        biases = [np.zeros(biases.shape) for biases in self._biases]
        weights = [np.zeros(weights.shape) for weights in self._weights]
        delta = (activations[-1] - y) * self._layers[-1].activation_function.prime(results[-1])
        biases[-1] = delta
        weights[-1] = np.dot(delta, activations[-2].T)
        for layer in xrange(2, len(self._layers)):
            delta = np.dot(self._weights[-layer + 1].T, delta) * \
                    self._layers[-layer].activation_function.prime(results[-layer])
            delta *= dropout_masks[-layer + 1]
            biases[-layer] = delta
            weights[-layer] = np.dot(delta, activations[-layer - 1].T)
        return biases, weights

    def _dropout(self, x, keep_probability):
        return np.random.RandomState().binomial(1, keep_probability, x.shape)

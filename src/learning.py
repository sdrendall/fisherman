__author__ = 'Sam Rendall'

import theano
from theano import tensor

class Model(object):
    """
    The base class for a Neural Network style model that is trained to fit an identification function
    This is a linked list of layers
    """

    def __init__(self, params=None, lam=0.0):
        """
        Params is a sequence of weight vectors corresponding to each layer of the model
        """
        self.params = params
        self.layers = []
        self.lam = lam  # TODO: What to call lambda?

    def set_lambda(self, lam):
        self.lam = lam

    def add_layer(self, layer, index):
        """
        Adds a layer to the model at the specified position
        """
        if index == 0:
            self.prepend_layer(layer)

        elif index == len(self.layers):
            self.append_layer(layer)

        else:
            self.layers.insert(index, layer)
            layer.prev_layer = self.layers[index - 1]
            layer.next_layer = self.layers[index + 1]
            self.layers[index - 1].next_layer = layer
            self.layers[index + 1].prev_layer = layer

    def remove_layer(self, index):
        """
        Removes the specified layer from the model
        """

        # Unlink the layer from the surrounding layers, if they exist
        layer_to_remove = self.layers[index]
        if layer_to_remove.next_layer is not None:
            layer_to_remove.next_layer.prev_layer = layer_to_remove.prev_layer

        if layer_to_remove.prev_layer is not None:
            layer_to_remove.prev_layer.next_layer = layer_to_remove.next_layer

        self.layers.pop(index)

    def append_layer(self, layer):
        """
        Appends a layer to the model
        """
        if len(self.layers) > 0:
            layer.prev_layer = self.layers[-1]
            self.layers[-1].next_layer = layer

        self.layers.append(layer)

    def prepend_layer(self, layer):
        """
        Prepends a layer to the model
        """
        if len(self.layers) > 0:
            layer.next_layer = self.layers[0]
            self.layers[0].prev_layer = layer

        self.layers.insert(0, layer)

    def train(self, inputs, outputs):
        """
        Trains the model on the given inputs and outputs
        """
        pass

    def evaluate(self, input):
        """
        Evaluates the model for the given input
        """
        # Raise error if self.params == None
        # Return 1 or 0
        pass

    def compute_gradient(self):
        """
        Computes the gradients for the parameters for each layer in the model
        Returns a vector of gradients that can be fed to an optimizer
        """
        pass


class Layer(object):
    """
    The base class for a layer of neurons in the model
    Each layer's parameters describe what they do with the activations from the previous layer

    Each layer computes an activation matrix A corresponding to the activation of each 'neuron' in that layer.
    """
    def __init__(self, size=None):
        self.size = size
        self.next_layer = None
        self.prev_layer = None

    def set_params(self, params):
        """
        Sets self.params with the given params, ensuring that they are valid
        :param params:
        """
        if params.size != self.get_number_of_params():
            raise InputError

        self.params = params
        self.Theta = theano.shared(self.params, 'Theta', borrow=True)

    def get_number_of_params(self):
        return self.size * (self.prev_layer.size + 1)


class InputLayer(Layer):
    """
    A fully connected input layer
    """

    def compose_forwards_prop(self):
        # Initialize Symbolic Variables
        self.X = tensor.matrix
        self.A = self.X

class HiddenLayer(Layer):
    """
    A hidden layer for the model

    This layer is fully connected to the previous layer
    A hidden layer is neither the input or output layer of a network
     and thus can safely assume the existence of a next and previous layer
    """
    def __init__(self, *args, **kwargs):
        Layer.__init__(self, *args, **kwargs)

        self.A = tensor.matrix('A')
        self.Z = tensor.matrix('Z')
        self.Delta = tensor.matrix('Delta')
        self.Gradient = tensor.matrix('Gradient')

    def compose_forward_prop(self):
        """
        The activation of this layer is based on the activation of the previous layer
        """
        self.Z = tensor.dot(self.prev_layer.A, self.Theta.T)
        self.A = tensor_sigmoid(self.Z)

    def compose_backwards_prop(self):
        """
        Creates an expression for the error of the previous layer based on this layer's error
        Creates an expression for this layer's Gradient
        """
        self.prev_layer.Delta = tensor.dot(self.Delta, self.Theta[:, 1:]) * tensor_sigmoid_gradient(self.Z)
        self.Gradient = theano.dot(self.Delta, self.prev_layer.A.T)

def tensor_sigmoid(z):
    """
    The sigmoid function, designed to work on theano tensors
    """
    return 1.0/(1.0 - tensor.exp(-z))


def tensor_sigmoid_gradient(z):
    """
    The gradient of the sigmoid function, designed to work on theano tensors
    """
    return tensor_sigmoid(z) * (1 - tensor_sigmoid(z))


class InputError(Exception):
    """
    Raised when insuffcient inputs are provided to a layer
    """
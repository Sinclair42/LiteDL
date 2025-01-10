"""
layer.py

This module defines a base class for neural network layers. The `Layer` class provides a template
for implementing custom layers in a neural network, including methods for forward and backward propagation.

Classes:
    Layer: A base class for defining layers in a neural network.
"""


class Layer(object):
    """
    A base class for neural network layers.

    This class serves as a template for defining custom layers in a neural network.
    Subclasses should override the `forward` and `backward` methods to implement the specific behavior
    of the layer.

    Attributes:
        params (dict or None): A dictionary to store layer-specific parameters such as weights and biases.
                                   Defaults to None for layers without trainable parameters.

    Methods:
        forward(x):
            Computes the forward pass of the layer. Must be overridden by subclasses.
            Args:
                x: Input data to the layer.
            Returns:
                Output data after applying the layer transformation.
            Raises:
                NotImplementedError: If the method is not implemented by the subclass.

        backward(dout):
            Computes the backward pass of the layer, including gradients with respect to its inputs.
            Must be overridden by subclasses.
            Args:
                dout: Gradient of the loss with respect to the output of this layer.
            Returns:
                Gradient of the loss with respect to the input of this layer.
            Raises:
                NotImplementedError: If the method is not implemented by the subclass.
        """
    def __init__(self):
        """
        Initializes the Layer object.

        The `params` attribute is set to None by default and can be overridden in subclasses
        if the layer has trainable parameters.
        """
        self.params = None

    def forward(self, x):
        """
        Performs the forward pass of the layer.

        This method should be implemented by subclasses to define how the input data is
        transformed by the layer.

        Args:
            x: Input data to the layer.

        Returns:
            The transformed output data.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
                """
        raise NotImplementedError

    def backward(self, dout):
        """
        Performs the backward pass of the layer.

        This method should be implemented by subclasses to compute gradients and propagate them
        to previous layers.

        Args:
            dout: Gradient of the loss with respect to the output of this layer.

        Returns:
            Gradient of the loss with respect to the input of this layer.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

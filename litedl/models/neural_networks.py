import pickle
import numpy as np
from litedl.core import Layer, Optimizer


class FeedForwardNeuralNetwork:
    """
    A feed-forward neural network implementation.

    This class allows for constructing and training feed-forward neural networks
    with customizable layers and loss functions.

    Attributes:
        layers (list): List of layers in the neural network.
        loss_layers (Layer): Loss layer for computing the loss and its gradient.
    """
    def __init__(self):
        """
        Initializes an empty neural network.
        """
        self.layers = []
        self.loss_layers = None

    def add_layer(self, layer):
        """
        Adds a layer to the neural network.

        Args:
            layer (Layer): An instance of a layer to be added.

        Raises:
            AssertionError: If `layer` is not a subclass of `Layer`.
        """
        assert issubclass(layer.__class__, Layer)

        self.layers.append(layer)

    def add_loss_layer(self, layer):
        """
        Sets the loss layer of the neural network.

        Args:
            layer (Layer): An instance of a loss layer.

        Raises:
            AssertionError: If `layer` is not a subclass of `Layer`.
        """
        assert issubclass(layer.__class__, Layer)

        self.loss_layers = layer

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the network to compute predictions.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Output of the neural network.
        """
        output = x
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def forward(self, x: np.ndarray, t: np.ndarray) -> float:
        """
        Performs a forward pass through the network and computes the loss.

        Args:
            x (np.ndarray): Input data.
            t (np.ndarray): Target labels.

        Returns:
            float: Computed loss.
        """
        output = self.predict(x)
        output = self.loss_layers.forward(output, t)

        return output

    def backward(self, dout=1.0):
        """
        Performs a backward pass through the network to compute gradients.

        Args:
            dout (float, optional): Gradient of the loss with respect to the output. Defaults to 1.0.
        """
        dout = self.loss_layers.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def step(self, optimizer):
        """
        Updates the parameters of the network using the optimizer.

        Args:
            optimizer (Optimizer): An instance of an optimizer.

        Raises:
            AssertionError: If `optimizer` is not a subclass of `Optimizer`.
        """
        assert issubclass(optimizer.__class__, Optimizer)

        for layer in self.layers:
            if layer.params is not None:
                optimizer.update(layer.params, layer.grads)

    @classmethod
    def from_pickle(cls, path):
        """
        Loads a saved neural network from a pickle file.

        Args:
            path (str): Path to the pickle file.

        Returns:
            FeedForwardNeuralNetwork: The loaded neural network.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

            return data

    def to_pickle(self, path):
        """
        Saves the neural network to a pickle file.

        Args:
            path (str): Path to save the pickle file.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def __repr__(self):
        """
        Returns a string representation of the neural network, including its layers.

        Returns:
            str: A string listing the layers of the network.
        """
        return f'''
{self.__class__.__name__}
        
layers:
{'\n'.join(str(layer) for layer in self.layers)}
        '''

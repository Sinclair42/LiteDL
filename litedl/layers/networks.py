from typing import Dict
import numpy as np
from litedl.core import Layer


class Affine(Layer):
    """
    Affine (fully connected) layer for neural networks.

    Attributes:
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        params (Dict[str, np.ndarray]): Dictionary containing weights ('W') and biases ('b').
        grads (Dict[str, np.ndarray]): Dictionary containing gradients for weights and biases.
    """
    def __init__(self, input_size: int, output_size: int, params: Dict[str, np.ndarray] | None = None):
        """
        Initializes the Affine layer with input size, output size, and optional parameters.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            params (Dict[str, np.ndarray] | None): Dictionary containing weights ('W') and biases ('b').
                If None, parameters are initialized randomly.

        Raises:
            ValueError: If the shapes of provided 'W' and 'b' do not match the expected dimensions.
        """
        self.x = None
        self.input_size = input_size
        self.output_size = output_size
        super().__init__()
        if params is None:
            self.params = {
                'W': np.random.randn(self.input_size, self.output_size),
                'b': np.random.randn(self.output_size),
            }
        else:
            if params['W'].shape != (self.input_size, self.output_size) or params['b'].shape != (self.output_size,):
                raise ValueError('W and b must have same shape')
            self.params = params
        self.grads = {
            'W': np.zeros_like(self.params['W']),
            'b': np.zeros_like(self.params['b'])
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass of the affine layer.

        Args:
            x (np.ndarray): Input data of shape (batch_size, input_size).

        Returns:
            np.ndarray: Output data of shape (batch_size, output_size).

        Raises:
            TypeError: If `x` is not a numpy array.
            ValueError: If `x` shape does not match `input_size`.
        """
        if type(x) is not np.ndarray:
            raise TypeError('x must be a numpy array')
        if x.shape[1] != self.input_size:
            raise ValueError('x shape must be same with input_size')

        self.x = x
        y = np.dot(x, self.params['W']) + self.params['b']

        return y

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass of the affine layer.

        Args:
            dout (np.ndarray): Gradient of loss with respect to the output, of shape (batch_size, output_size).

        Returns:
            np.ndarray: Gradient of loss with respect to the input, of shape (batch_size, input_size).

        Raises:
            TypeError: If `dout` is not a numpy array.
            ValueError: If `dout` shape does not match `output_size`.
        """
        if type(dout) is not np.ndarray:
            raise TypeError('dout must be a numpy array')
        if dout.shape[1] != self.output_size:
            raise ValueError('dout shape must be same with output_size')

        dx = np.dot(dout, self.params['W'].T)
        self.grads['W'] = np.dot(self.x.T, dout)
        self.grads['b'] = np.sum(dout, axis=0)

        return dx

    @classmethod
    def from_dict(cls, params: Dict[str, np.ndarray]):
        """
        Create an Affine layer from a dictionary of parameters.

        Args:
            params (Dict[str, np.ndarray]): Dictionary containing weights ('W') and biases ('b').

        Returns:
            Affine: An initialized Affine layer.
        """
        input_size = params['W'].shape[0]
        output_size = params['W'].shape[1]
        return cls(input_size, output_size, params)

    def to_dict(self):
        """
        Export the parameters of the layer as a dictionary.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing weights ('W') and biases ('b').
        """
        return self.params

    def __repr__(self):
        """
        Print the name of class

        Returns:
            str: Affine()
        """
        return f'{self.__class__.__name__}()'

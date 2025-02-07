import numpy as np
from litedl.core import Layer


class Sigmoid(Layer):
    """
    Sigmoid activation layer for neural networks.

    The Sigmoid function outputs values between 0 and 1, making it suitable for probabilistic interpretations.

    Attributes:
        y (np.ndarray): The output of the sigmoid function during the forward pass.
    """
    def __init__(self):
        """
        Initializes the Sigmoid layer.

        Inherits from the base Layer class.
        """
        super().__init__()
        self.y = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass using the sigmoid function.

        Args:
            x (np.ndarray): Input array for the layer.

        Returns:
            np.ndarray: The result of applying the sigmoid function to the input.
        """
        y = 1 / (1 + np.exp(-1 * x))
        self.y = y

        return y

    def backward(self, dout):
        """
        Performs the backward pass to compute the gradient of the loss with respect to the input.

        Args:
            dout (np.ndarray): Gradient of the loss with respect to the output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        dx = dout * (1.0 - self.y) * self.y

        return dx

    def __repr__(self):
        """
        Returns a string representation of the Sigmoid layer.

        Returns:
            str: A string in the format "Sigmoid()".
        """
        return f'{self.__class__.__name__}()'


class ReLU(Layer):
    """
    ReLU (Rectified Linear Unit) activation layer for neural networks.

    This layer applies the ReLU activation function element-wise, which replaces negative values with zero.

    Attributes:
        mask (np.ndarray): A boolean mask indicating where input values are less than or equal to zero.
    """
    def __init__(self):
        """
        Initializes the ReLU layer.
        """
        super().__init__()
        self.mask = None

    def forward(self, x: np.ndarray):
        """
        Perform the forward pass of the ReLU activation function.

        Args:
            x (np.ndarray): Input data of any shape.

        Returns:
            np.ndarray: Output data where all negative values in `x` are replaced with zero.
        """
        self.mask = (x <= 0)
        y = x.copy()
        y[self.mask] = 0

        return y

    def backward(self, dout):
        """
        Perform the backward pass of the ReLU activation function.

        Args:
            dout (np.ndarray): Gradient of the loss with respect to the output of the layer.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of the layer.
        """
        dx = dout.copy()
        dx[self.mask] = 0

        return dx

    def __repr__(self):
        """
        Return a string representation of the ReLU layer.

        Returns:
            str: A string in the format "ReLU()".
        """
        return f'{self.__class__.__name__}()'

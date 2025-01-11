import numpy as np
from litedl.core import Layer


class MSE(Layer):
    """
    Mean Squared Error (MSE) loss layer.

    This class calculates the mean squared error between the predicted values (y)
    and the target values (t), which is commonly used as a loss function in regression tasks.

    Attributes:
        n (int): Number of samples in the input batch.
        y (np.ndarray): Predicted values.
        t (np.ndarray): Target values.
    """
    def __init__(self):
        """
        Initializes the MSE loss layer.
        """
        super().__init__()
        self.n = None
        self.y = None
        self.t = None

    def forward(self, y: np.ndarray, t: np.ndarray):
        """
        Performs the forward pass to compute the MSE loss.

        Args:
            y (np.ndarray): Predicted values of shape (batch_size, ...).
            t (np.ndarray): Target values of shape (batch_size, ...).

        Returns:
            float: The mean squared error loss.

        Notes:
            The loss is computed as the average of the squared differences
            between the predicted and target values:
                loss = (1/n) * sum((y - t)^2)
        """

        self.n = y.shape[0]
        self.y = y
        self.t = t
        loss = np.sum((y - t) ** 2) / self.n

        return loss

    def backward(self, dout=1.0):
        """
        Performs the backward pass to compute the gradient of the loss
        with respect to the input values (y).

        Args:
            dout (float, optional): Upstream gradient. Defaults to 1.0.

        Returns:
            np.ndarray: Gradient of the loss with respect to the predicted values (y),
                        of the same shape as y.

        Notes:
            The gradient is computed as:
                dL/dy = (2/n) * (y - t) * dout
        """
        dout = dout * np.ones_like(self.y)
        dout = dout / self.n
        dout = dout * 2 * (self.y - self.t)

        return dout

    def __repr__(self):
        """
        Returns a string representation of the MSE layer.

        Returns:
            str: A string in the format "MSE()".
        """
        return f'{self.__class__.__name__}()'

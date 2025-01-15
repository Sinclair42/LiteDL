import numpy as np
from litedl.core import Layer
from litedl.core.function import softmax, cross_entropy_error


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


class SoftmaxWithLoss(Layer):
    """
    A layer that combines the softmax activation function with the cross-entropy loss.
    This layer is typically used as the final layer in a classification neural network.

    Attributes:
        y (np.ndarray or None): The output probabilities from the softmax function.
        t (np.ndarray or None): The ground truth labels.
    """
    def __init__(self):
        """
        Initializes the SoftmaxWithLoss layer.
        """

        super().__init__()
        self.y = None
        self.t = None

    def forward(self, x: np.ndarray, t: np.ndarray):
        """
        Performs the forward pass by calculating the softmax probabilities and the cross-entropy loss.

        Args:
            x (np.ndarray): Input data (logits), of shape (batch_size, num_classes).
            t (np.ndarray): Ground truth labels, of shape (batch_size, num_classes) or (batch_size,).

        Returns:
            float: The calculated cross-entropy loss.
        """
        self.t = t
        self.y = softmax(x)
        loss = cross_entropy_error(self.y, self.t)

        return loss

    def backward(self, dout=1.0):
        """
        Performs the backward pass, calculating the gradient of the loss with respect to the input.

        Args:
            dout (float): Upstream gradient (default is 1.0).

        Returns:
            np.ndarray: The gradient of the loss with respect to the input, of shape (batch_size, num_classes).
        """
        batch_size = self.t.shape[0]
        dx = dout * (self.y - self.t) / batch_size

        return dx

    def __repr__(self):
        """
        Returns a string representation of the layer.

        Returns:
            str: A string in the format "SoftmaxWithLoss()".
        """
        return f'{self.__class__.__name__}()'

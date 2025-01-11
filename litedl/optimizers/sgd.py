from typing import Dict
import numpy as np
from litedl.core import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.

    This class implements the SGD optimization algorithm to update model parameters
    based on their gradients and a specified learning rate.

    Attributes:
        lr (float): Learning rate for the optimizer.
    """
    def __init__(self, lr=0.01):
        """
        Initializes the SGD optimizer.

        Args:
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.01.
        """
        super().__init__(lr)

    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """
        Updates parameters using their gradients.

        Args:
            params (dict): Dictionary of model parameters (e.g., weights and biases).
                           Each value should be a NumPy array.
            grads (dict): Dictionary of gradients corresponding to each parameter in `params`.
                          The keys in `grads` must match those in `params`.

        Notes:
            Each parameter is updated as:
                param = param - lr * grad
        """
        for key, value in params.items():
            params[key] -= self.lr * grads[key]

    def __repr__(self) -> str:
        """
        Returns a string representation of the SGD optimizer.

        Returns:
            str: A string in the format "SGD(lr=learning_rate)".
        """
        return f'{self.__class__.__name__}(lr={self.lr})'

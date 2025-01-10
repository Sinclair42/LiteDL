"""
optimizer.py

This module defines a base class for optimization algorithms in neural networks.
The `Optimizer` class provides a template for implementing custom optimization algorithms
to update model parameters based on gradients.

Classes:
    Optimizer: A base class for optimization algorithms.
"""


class Optimizer(object):
    """
    A base class for optimization algorithms in neural networks.

    This class serves as a template for defining custom optimizers used in training neural networks.
    Subclasses should override the `update` method to implement the specific optimization logic.

    Attributes:
        lr (float): Learning rate for the optimization algorithm.

    Methods:
        update(params, grads):
            Updates the model parameters based on their gradients and the optimization logic.
            Must be overridden by subclasses.
            Args:
                params (dict): Dictionary containing the model parameters (e.g., weights and biases).
                grads (dict): Dictionary containing the gradients of the model parameters.
            Returns:
                None
            Raises:
                NotImplementedError: If the method is not implemented by the subclass.
    """
    def __init__(self, lr):
        """
        Initializes the Optimizer object with a learning rate.

        Args:
            lr (float): The learning rate for the optimization algorithm.
        """
        self.lr = lr

    def update(self, params, grads):
        """
        Updates the model parameters using the specified optimization logic.

        This method should be implemented by subclasses to define how parameters
        are updated based on their gradients and the learning rate.

        Args:
            params (dict): Dictionary containing the model parameters (e.g., weights and biases).
            grads (dict): Dictionary containing the gradients of the model parameters.

        Returns:
            None

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

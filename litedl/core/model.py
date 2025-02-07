import pickle

import numpy as np


class BaseModel:
    """
    A base class for machine learning models.

    This class defines the fundamental structure of a model, including methods for
    prediction, forward propagation, backward propagation, optimization step updates,
    and serialization.

    Methods:
        predict(x): Should be implemented in subclasses to make predictions.
        forward(x): Should be implemented in subclasses to define forward propagation.
        backward(dout): Should be implemented in subclasses to define backward propagation.
        step(optimizer): Should be implemented in subclasses to update parameters.
        from_pickle(path): Loads a saved model from a pickle file.
        to_pickle(path): Saves the model to a pickle file.
    """

    def __init__(self):
        """
        Initializes the BaseModel.
        """
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes a prediction based on input data.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Model predictions.

        Raises:
            NotImplementedError: Must be implemented in a subclass.
        """
        raise NotImplementedError

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs forward propagation.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: The processed output.

        Raises:
            NotImplementedError: Must be implemented in a subclass.
        """
        raise NotImplementedError

    def backward(self, dout=1.0):
        """
        Performs backward propagation.

        Args:
            dout (float, optional): The upstream gradient. Defaults to 1.0.

        Raises:
            NotImplementedError: Must be implemented in a subclass.
        """
        raise NotImplementedError

    def step(self, optimizer):
        """
        Updates model parameters using the provided optimizer.

        Args:
            optimizer: The optimizer instance used for updating model parameters.

        Raises:
            NotImplementedError: Must be implemented in a subclass.
        """
        raise NotImplementedError

    @classmethod
    def from_pickle(cls, path):
        """
        Loads a model from a pickle file.

        Args:
            path (str): Path to the pickle file.

        Returns:
            BaseModel: The deserialized model instance.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

            return data

    def to_pickle(self, path):
        """
        Saves the model to a pickle file.

        Args:
            path (str): Path where the model should be saved.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def __repr__(self):
        """
        Returns a string representation of the model.

        Returns:
            str: A string in the format "BaseModel()".
        """
        return f'{self.__class__.__name__}()'

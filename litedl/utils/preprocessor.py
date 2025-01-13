import numpy as np


class Standardizer:
    """
    A class to standardize data by removing the mean and scaling to unit variance.

    Attributes:
        mean (np.ndarray or None): The mean of the data along each feature, calculated during fitting.
        std (np.ndarray or None): The standard deviation of the data along each feature, calculated during fitting.
    """
    def __init__(self):
        """
        Initializes the Standardizer with mean and standard deviation set to None.
        """
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        """
        Calculate the mean and standard deviation of the data.

        Args:
            data (np.ndarray): The data to fit, with shape (n_samples, n_features).
        """
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)

    def transform(self, data: np.ndarray):
        """
        Standardize the data using the calculated mean and standard deviation.

        Args:
            data (np.ndarray): The data to transform, with shape (n_samples, n_features).

        Returns:
            np.ndarray: The standardized data.

        Raises:
            AssertionError: If the standardizer has not been fitted before transformation.
        """
        assert self.mean is not None and self.std is not None

        return (data - self.mean) / self.std


class MinMaxScaler:
    """
    A class to scale data to a specified range (default 0 to 1).

    Attributes:
        max (np.ndarray or None): The maximum value for each feature, calculated during fitting.
        min (np.ndarray or None): The minimum value for each feature, calculated during fitting.
    """
    def __init__(self):
        """
        Initializes the MinMaxScaler with min and max set to None.
        """
        self.max = None
        self.min = None

    def fit(self, data: np.ndarray):
        """
        Calculate the minimum and maximum values of the data.

        Args:
            data (np.ndarray): The data to fit, with shape (n_samples, n_features).
        """
        self.max = data.max(axis=0)
        self.min = data.min(axis=0)

    def transform(self, data: np.ndarray):
        """
        Scale the data to the range [0, 1] using the calculated min and max values.

        Args:
            data (np.ndarray): The data to transform, with shape (n_samples, n_features).

        Returns:
            np.ndarray: The scaled data.

        Raises:
            AssertionError: If the scaler has not been fitted before transformation.
        """
        assert self.max is not None and self.min is not None

        return (data - self.min) / (self.max - self.min)


class OneHotEncoder:
    """
    A class to perform one-hot encoding on categorical data.

    Attributes:
        index (Dict or None): A dictionary mapping unique categorical values to their corresponding indices.
    """
    def __init__(self):
        """
        Initializes the OneHotEncoder with the index set to None.
        """
        self.index = None

    def fit(self, data: np.ndarray):
        """
        Learn the unique categorical values from the data and create an index mapping.

        Args:
            data (np.ndarray): The data to fit, with shape (n_samples, 1).

        Raises:
            AssertionError: If the input data is not a 2D array with a single column.
        """
        assert data.ndim == 2 and data.shape[1] == 1

        result = data.copy()
        result = result.flatten()
        result = np.unique(result)

        self.index = {word: index for index, word in enumerate(result)}

    def transform(self, data: np.ndarray):
        """
        Transform the data into one-hot encoded vectors.

        Args:
            data (np.ndarray): The data to transform, with shape (n_samples, 1).

        Returns:
            np.ndarray: One-hot encoded representation of the data, with shape (n_samples, n_categories).

        Raises: AssertionError: If the encoder has not been fitted or if the input data is not a 2D array with a
        single column.
        """
        assert self.index is not None
        assert data.ndim == 2 and data.shape[1] == 1

        one_hot_vectors = []
        result = data.copy().flatten()

        for word in result:
            vector = [0] * len(self.index)
            if word in self.index:
                vector[self.index[word]] = 1
            one_hot_vectors.append(vector)

        one_hot_vectors = np.array(one_hot_vectors)

        return one_hot_vectors

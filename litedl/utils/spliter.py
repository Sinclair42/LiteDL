import numpy as np


def data_split(data: np.ndarray, rate: float):
    """
    Split the dataset into training and testing subsets.

    Args:
        data (np.ndarray): The input dataset to be split, with shape (n_samples, n_features).
        rate (float): The proportion of the dataset to use as the test set. Must be between 0 and 1.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - train_features (np.ndarray): Training data, with shape (n_train_samples, n_features).
            - test_features (np.ndarray): Testing data, with shape (n_test_samples, n_features).

    Raises:
        AssertionError: If the rate is not between 0 and 1.

    Example:
        >>> data1 = np.arange(10).reshape(10, 1)
        >>> train, test = data_split(data1, 0.2)
        >>> print(train)
        [[0]
         [1]
         [2]
         [3]
         [4]
         [5]
         [6]
         [7]]
        >>> print(test)
        [[8]
         [9]]
    """
    assert 0 <= rate <= 1

    data_size = int(data.shape[0] * (1 - rate))

    train_features = data[:data_size]
    test_features = data[data_size:]

    return train_features, test_features


def get_batches(features: np.ndarray, batch_size: int):
    """
    Divide the dataset into batches of a specified size.

    Args:
        features (np.ndarray): The dataset to be divided, with shape (n_samples, n_features).
        batch_size (int): The size of each batch.

    Returns: List[np.ndarray]: A list of batches, where each batch is a numpy array of shape (batch_size, 
    n_features). The last batch may have fewer samples if the total number of samples is not divisible by the batch 
    size.

    Example:
        >>> features1 = np.arange(10).reshape(10, 1)
        >>> batches = get_batches(features1, 3)
        >>> for batch in batches:
        ...     print(batch)
        [[0]
         [1]
         [2]]
        [[3]
         [4]
         [5]]
        [[6]
         [7]
         [8]]
    """
    index = np.random.permutation(features.shape[0])
    batche_data = []

    for i in range(0, features.shape[0]//batch_size):
        batche_data.append(features[index[i * batch_size:(i + 1) * batch_size], :])

    return batche_data

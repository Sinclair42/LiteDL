import numpy as np


def softmax(x: np.ndarray):
    c = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    y = exp_x / sum_exp_x

    return y


def cross_entropy_error(y: np.ndarray, t: np.ndarray):
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

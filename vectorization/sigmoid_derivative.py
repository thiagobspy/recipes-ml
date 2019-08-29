import numpy as np


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))

    return s


def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s * (1 - s)

    return ds


x = np.array([1, 2, 3])
print("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))

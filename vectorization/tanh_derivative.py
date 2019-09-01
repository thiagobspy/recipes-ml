import numpy as np


def tanh(x):
    x_exp = np.exp(x)
    x1_exp = np.exp(-x)

    return (x_exp - x1_exp) / (x_exp + x1_exp)


def tanh_derivative(x):
    return 1 - tanh(x) ** 2

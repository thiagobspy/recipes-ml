import numpy as np


def softmax(x):
    x_exp = np.exp(x)

    x_sum = np.sum(x_exp, axis=1, keepdims=True)

    s = x_exp / x_sum

    return s


x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0, 0]])

print(x.shape)

print("softmax(x) = " + str(softmax(x)))

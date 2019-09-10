import numpy as np


def norm_mean(x):
    return x - np.mean(x)


y = np.array([10, 20, 30, 0, 6, 4])

print(norm_mean(y))

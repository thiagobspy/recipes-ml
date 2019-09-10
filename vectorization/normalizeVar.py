import numpy as np


def norm_var(x):
    return x / np.var(x)


y = np.array([10, 20, 30, 0, 6, 4])

print(norm_var(y))

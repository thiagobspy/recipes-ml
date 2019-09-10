import numpy as np


# The L2 norm that is calculated as the square root of the sum of the squared vector values.

def L2(x):
    loss = np.sqrt(sum(x ** 2))

    return loss


yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat - y)))

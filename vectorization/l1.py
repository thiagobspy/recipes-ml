import numpy as np


# The L1 norm that is calculated as the sum of the absolute values of the vector.

def L1(x):
    loss = sum(abs(x))

    return loss


yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat - y)))

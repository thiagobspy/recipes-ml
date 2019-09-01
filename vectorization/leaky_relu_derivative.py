def leaky_relu(x):
    return max(0.01 * x, x)


def leaky_relu_derivative(x):
    if x >= 0:
        return 1
    else:
        return 0.01

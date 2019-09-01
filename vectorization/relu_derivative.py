def relu(x):
    return max(0, x)


def relu_derivative(x):
    if x >= 0:
        return 1
    else:
        return 0

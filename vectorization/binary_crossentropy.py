import numpy


# m = n sample
# Y = target correct
# A = predict
def logistic(m, Y, A):
    return (-1 / m) * numpy.sum(Y * numpy.log(A) + (1 - Y) * numpy.log(1 - A))


def logistic2(Y, A):
    return -numpy.dot(Y, numpy.log(A).T)

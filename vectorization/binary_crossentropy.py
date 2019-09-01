import numpy


# m = n sample
# Y = target correct
# A = predict
def logistic(m, Y, A):
    cost = (-1 / m) * numpy.sum(Y * numpy.log(A) + (1 - Y) * numpy.log(1 - A))

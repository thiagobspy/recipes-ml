import numpy

# input = 10
# output =5
shape = (10, 5)


# quando relu a constante deve ser 2
# quando tanh ou sigmoid a constante deve ser 1

def xavier_init(shape):
    return numpy.random.randn(shape) * numpy.sqrt(2 / (shape[0] + shape[1]))

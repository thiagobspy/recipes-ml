import numpy


def sigmoid(z):
    return 1 / (1 + numpy.exp(-z))


# create dataset
# Podemos optar com X random ou fixo
X = numpy.random.randint(0, 5, size=(10, 3))
X = numpy.asarray([[0, 0, 2],
                   [0, 3, 0],
                   [4, 3, 0],
                   [3, 1, 2],
                   [0, 4, 0],
                   [4, 3, 1],
                   [3, 4, 3],
                   [0, 4, 0],
                   [1, 3, 0],
                   [4, 1, 2]])
Y = numpy.asarray([1 for x in X])
print(X)
print(Y)

# init weight
# Podemos optar por random ou peso fixo
w = (numpy.random.rand(X.shape[1], 1) - 0.5) * 0.1
b = (numpy.random.rand(1, 1) - 0.5) * 0.1

w = numpy.asarray([[0.1], [-0.1], [0]])
b = numpy.asarray([0])
print(w)
print(b)

learning = 0.04
epochs = 10

for i in range(epochs):
    z = X.dot(w) + b
    yhat = sigmoid(z)

    Y = Y.reshape(Y.shape[0], 1)
    diff = yhat - Y

    cost_mean_squared = numpy.sum((yhat - Y) ** 2 / 2) / yhat.shape[0]
    cost_log = numpy.sum(- (Y * numpy.log(yhat) + (1 - Y) * numpy.log(1 - yhat))) / yhat.shape[0]

    dw = diff.transpose().dot(X) / yhat.shape[0]
    db = numpy.sum(diff) / yhat.shape[0]

    w = w - learning * dw.transpose()
    b = b - learning * db

    print('Epochs: ', i)
    print('mean: ', cost_mean_squared)
    print('cost: ', cost_log)
    print('New weight: \n', w)
    print('New bias:', b)
    print('accuracy: ', numpy.sum((yhat >= 0.5) == Y) / len(yhat))
    print('=' * 20)

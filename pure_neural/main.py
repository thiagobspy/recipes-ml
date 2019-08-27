import numpy

def sigmoid(z):
    return 1 / (1 + numpy.exp(-z))

# create dataset
X = numpy.random.randint(0, 10, size=(10,3))
Y = numpy.asarray([int(x[-1] % 2 == 0) for x in X])

print(X)
print(Y)

# init weight
w = numpy.random.rand(3, 1)
print(w)

z = X.dot(w)
print(z)
yhat = sigmoid(z)
print(yhat)

Y = Y.reshape(10, 1)
diff = yhat-Y
print(diff)

cost_mean_squared = numpy.sum((yhat - Y) ** 2 / 2) / yhat.shape[0]
print(cost_mean_squared)

cost_log = numpy.sum(- (Y * numpy.log(yhat) + (1 - Y) * numpy.log(1 - yhat))) / yhat.shape[0]
print(cost_log)

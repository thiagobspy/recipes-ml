import matplotlib.pyplot as plt

import numpy
import h5py


def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = numpy.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = numpy.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = numpy.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = numpy.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = numpy.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# Input: [10,5,3,1]
# Output: dict of weight/bias
def init_weight_deep(layer_dim):
    weight = {}
    L = len(layer_dim)

    for i in range(1, L):
        weight['W' + str(i)] = numpy.random.randn(layer_dim[i], layer_dim[i - 1]) * 0.01
        weight['b' + str(i)] = numpy.zeros((layer_dim[i], 1))

        assert (weight['W' + str(i)].shape == (layer_dim[i], layer_dim[i - 1]))
        assert (weight['b' + str(i)].shape == (layer_dim[i], 1))

    return weight


# Input: A or X, Weight, bias
# Output: Z (only linear) and cache for back propagation
def linear_forward(A, W, b):
    Z = numpy.dot(W, A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


# Input: A or X, Weight, bias and activation function
# Output: A_actual and cache [A_prev, W, b, Z]
def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    A = activation(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    cache = (linear_cache, Z)

    return A, cache


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def relu(x):
    return numpy.maximum(0, x)


# Input: X [A0], weights
# Output: Y, list of caches
def l_model_forward(X, weights):
    caches = []
    A = X
    L = len(weights) // 2

    for l in range(1, L):
        A, cache = linear_activation_forward(A, weights['W' + str(l)], weights['b' + str(l)], relu)
        caches.append(cache)

    A, cache = linear_activation_forward(A, weights['W' + str(L)], weights['b' + str(L)], sigmoid)
    caches.append(cache)

    assert (A.shape == (1, X.shape[1]))
    assert (A.shape == (weights['W' + str(L)].shape[0], X.shape[1]))

    return A, caches


def cost_function(Yhat, Y):
    m = Y.shape[1]
    cost = -(1 / m) * numpy.sum(numpy.dot(Y, numpy.log(Yhat).T) + numpy.dot((1 - Y), numpy.log(1 - Yhat).T))

    # flat
    cost = numpy.squeeze(cost)
    assert (cost.shape == ())

    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1 / m * numpy.dot(dZ, A_prev.T)
    db = 1 / m * numpy.sum(dZ, axis=1, keepdims=True)
    dA_prev = numpy.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def relu_derivative(x, activation):
    return x * numpy.where(activation > 0, 1, 0)


def sigmoid_derivative(x, activation):
    return x * (sigmoid(activation) * (1 - sigmoid(activation)))


def linear_activation_backward(dA, cache, derivative):
    linear_cache, activation_cache = cache

    dZ = derivative(dA, activation_cache)
    return linear_backward(dZ, linear_cache)


def l_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (numpy.divide(Y, AL) - numpy.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                      current_cache,
                                                                                                      sigmoid_derivative)
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,
                                                                    relu_derivative)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(weights, grads, learning_rate):
    L = len(weights) // 2

    for l in range(L):
        weights["W" + str(l + 1)] = weights["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        weights["b" + str(l + 1)] = weights["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return weights


def predict(X, y, weights):
    m = X.shape[1]
    n = len(weights) // 2
    p = numpy.zeros((1, m))

    probas, caches = l_model_forward(X, weights)

    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    return p


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
index = 3
plt.imshow(train_x_orig[index])
print("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.")

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print("Number of training examples: " + str(m_train))
print("Number of testing examples: " + str(m_test))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_x_orig shape: " + str(train_x_orig.shape))
print("train_y shape: " + str(train_y.shape))
print("test_x_orig shape: " + str(test_x_orig.shape))
print("test_y shape: " + str(test_y.shape))

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))

n_x = 12288
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)


def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    grads = {}
    costs = []
    m = X.shape[1]

    parameters = init_weight_deep(layers_dims)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):

        A1, cache1 = linear_activation_forward(X, W1, b1, relu)
        A2, cache2 = linear_activation_forward(A1, W2, b2, sigmoid)

        cost = cost_function(A2, Y)

        dA2 = - (numpy.divide(Y, A2) - numpy.divide(1 - Y, 1 - A2))

        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, sigmoid_derivative)
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, relu_derivative)

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, numpy.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost

    plt.plot(numpy.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


parameters = two_layer_model(train_x, train_y, layers_dims=(n_x, n_h, n_y), num_iterations=2500, print_cost=True)

predictions_train = predict(train_x, train_y, parameters)
print("Accuracy: " + str(numpy.sum((predictions_train == train_y)) / train_x.shape[1]))

predictions_test = predict(test_x, test_y, parameters)
print("Accuracy: " + str(numpy.sum((predictions_test == test_y)) / test_x.shape[1]))

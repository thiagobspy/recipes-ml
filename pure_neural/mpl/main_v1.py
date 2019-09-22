import numpy
'''
Com camada de entrada e de saida apenas
'''
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
Y = Y.reshape(Y.shape[0], 1)
print(X)
print(Y)

learning = 0.04
epochs = 10


# Declaracao da funcao ativacao
def sigmoid(z):
    s = 1 / (1 + numpy.exp(-z))

    return s


# Declaracao da funcao para inicializar pesos e bias
def initialize_with_zeros(dim):
    w = numpy.zeros((dim, 1))
    b = 0.0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


def propagate(w, b, X, Y):
    m = X.shape[0]

    A = sigmoid(numpy.dot(X, w) + b)

    cost = (-1 / m) * numpy.sum(Y * numpy.log(A) + (1 - Y) * numpy.log(1 - A))

    dw = (1 / m) * numpy.dot(X.T, (A - Y))
    db = (1 / m) * numpy.sum(A - Y)

    cost = numpy.squeeze(cost)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    assert (cost.shape == ())

    grads = {"dw": dw, "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[0]
    y_prediction = numpy.zeros((1, m))
    w = w.reshape(X.shape[1], 1)

    A = sigmoid(numpy.dot(X, w) + b)
    y_prediction = (A >= 0.5).T

    assert (y_prediction.shape == (1, m))

    return y_prediction


def model(X_train, Y_train, X_test=None, Y_test=None, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[1])

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = parameters["w"]
    b = parameters["b"]

    y_prediction_train = predict(w, b, X_train)
    print("train accuracy: {} %".format(100 - numpy.mean(numpy.abs(y_prediction_train - Y_train)) * 100))

    y_prediction_test = None
    if X_test is not None and Y_test is not None:
        y_prediction_test = predict(w, b, X_test)
        print("test accuracy: {} %".format(100 - numpy.mean(numpy.abs(y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "y_prediction_test": y_prediction_test,
         "y_prediction_train": y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


m_train = X.shape[0]
m_var = X.shape[1]

# Normalizacao
X = X / numpy.amax(X)

# Resumo
print("Number of training examples: m_train = " + str(m_train))
print("train_set_x shape: " + str(X.shape))
print("train_set_y shape: " + str(Y.shape))

d = model(X, Y, num_iterations=2000, learning_rate=0.005, print_cost=True)

import numpy as np

"""
Formally, this will be the exponentially weighted average of the gradient on previous steps.
 You can also think of  vv  as the "velocity" of a ball rolling downhill,
  building up speed (and momentum) according to the direction of the gradient/slope of the hill.
  
  VdW = B*VdW + (1-B)*dW
  W = W - a * VdW
  
  Vdb = B*Vdb + (1-B)*db
  b = b - a * Vdb
  
B = momentun (normalmente 0.8 to 0.999)

O resto dos parametro sao iguais do batch_gradient.

O importante nota, que esse cenario, segue a ideia da media movel exponencial, onde o resultado anterior tem impacto no resultado atual,
evitando oscilacoes grandes.
"""


def initialize_velocity(parameters):
    L = len(parameters) // 2
    v = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros((parameters['W' + str(l + 1)].shape[0], parameters['W' + str(l + 1)].shape[1]))
        v["db" + str(l + 1)] = np.zeros((parameters['b' + str(l + 1)].shape[0], parameters['b' + str(l + 1)].shape[1]))

    return v


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)]

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]

    return parameters, v

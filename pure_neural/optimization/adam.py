import numpy as np

"""
How does Adam work?

It calculates an exponentially weighted average of past gradients, and stores it in variables  v  (before bias correction) and  v_correct  (with bias correction).
It calculates an exponentially weighted average of the squares of the past gradients, and stores it in variables  s  (before bias correction) and  s_correct  (with bias correction).
It updates parameters in a direction based on combining information from "1" and "2".

Agora sao 2 parametros recebidos (beta1 e beta2)

  VdW = B1*VdW + (1-B2)*dW
  VdW_correct = VdW/(1-B1**t)

  SdW = B1*SdW + (1-B2)*dW^2
  SdW_correct = SdW/(1-B2**t)

  W = W - a * (VdW_correct / sqrt(SdW_correct) + eplison)

As variaveis correct sao para corrigir o problema dos primeiros valores comecarem com 0, uma vez que estamos fazendo media ponderada.

t = counts the number of steps taken of Adam
ε = is a very small number to avoid dividing by zero

A logica do t, eh depois de algum tempo, nao precisa corrigir os valores calculado, entao as variaveis correct tende a serem iguais as valores calculado do v e s
"""


def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros((parameters['W' + str(l + 1)].shape[0], parameters['W' + str(l + 1)].shape[1]))
        v["db" + str(l + 1)] = np.zeros((parameters['b' + str(l + 1)].shape[0], parameters['b' + str(l + 1)].shape[1]))
        s["dW" + str(l + 1)] = np.zeros((parameters['W' + str(l + 1)].shape[0], parameters['W' + str(l + 1)].shape[1]))
        s["db" + str(l + 1)] = np.zeros((parameters['b' + str(l + 1)].shape[0], parameters['b' + str(l + 1)].shape[1]))

    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for l in range(L):
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]

        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - beta1 ** t)
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - beta1 ** t)

        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * grads["dW" + str(l + 1)] ** 2
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * grads["db" + str(l + 1)] ** 2

        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - beta2 ** t)
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - beta2 ** t)

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v_corrected["dW" + str(l + 1)] / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon)

    return parameters, v, s

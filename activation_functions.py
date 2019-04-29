import numpy as np


def ReLU(Z):
    return np.maximum(np.zeros_like(Z), Z)

def grad_ReLU(Z):
    grad = np.zeros_like(Z)
    grad[Z >= 0] = 1

    return grad

def softmax(Z):
    exps = np.exp(Z-np.max(Z, axis=1)[:, None])
    return exps/(np.sum(exps, axis=1)[:, None])

def grad_softmax(Z):
    pass

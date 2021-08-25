import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derv(x):
    return sigmoid(x)*(1-sigmoid(x))

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def tanh_derv(x):
    return (1 - tanh(x))**2

def relu(x):
    return np.maximum(0,x)

def relu_derv(x):
    return (x > 0).astype(int)

def leaky_relu(x):
    return np.maximum(0.01*x,0)

def leaky_relu_derv(x):
    dx = np.ones_like(x)
    dx[x < 0] = 0.01
    return dx
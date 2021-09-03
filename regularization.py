import numpy as np

def L2_regularization(m,w,l2_labda):
    regularized_loss = l2_labda/(2*m) * np.sum(np.square(w))
    return regularized_loss

def dropout_regularization(a,keep_prob=0.7):
    dropout_vector = np.random.rand(a.shape[0],a.shape[1])<keep_prob
    a = a * dropout_vector
    a = a/keep_prob
    return a

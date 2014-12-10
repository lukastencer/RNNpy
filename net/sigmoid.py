import numpy as np

def sigmoid(X):
    
    return 1 / ( 1 + np.exp( - X ) )

def sigmoidGrad(X):
    
    sig = sigmoid(X)
    
    return sig * (1-sig)
from sigmoid import *

def sigLin(params,X):
    
    return sigmoid(np.dot(X,params))
    
def sigLinGrad(params,X):
    h0g = sigmoidGrad(np.dot(X,params))
    if len(X.shape) > 1:
        return np.repeat(np.expand_dims(h0g,1),X.shape[1],1)*X
    else:
        return h0g*X 
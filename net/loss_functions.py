import numpy as np

def squaredLoss(trueY,h0):
    
    return np.sum(np.square(trueY - h0))

def squaredLossGrad(trueY,h0,h0grad):
    
    error = trueY-h0
    sqr = 2*np.repeat(np.expand_dims(error,1),h0grad.shape[1],1) * (-h0grad)
    
    return np.sum(sqr,0)

def logLoss(trueY,h0):
    
    cost = (-trueY * np.log(h0)) - ((1-trueY)*np.log(1-h0))
    return np.sum(cost)

def logLossGrad(trueY,h0,h0grad):
    
    constTerm1 = (-trueY*1/h0)
    term1 = np.repeat(np.expand_dims(constTerm1,1),h0grad.shape[1],1) * h0grad
    
    constTerm2 = ((1-trueY)*1/(1-h0))
    term2 =  np.repeat(np.expand_dims(constTerm2,1),h0grad.shape[1],1) * -h0grad
    grad = term1 - term2
    
    return np.sum(grad,0)
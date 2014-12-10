from scipy.optimize import check_grad
import numpy as np
from sigmoid import *
from hypothesis import *
from loss_functions import *

# h0 = sigmoid(np.dot(X,params.T))
# error = y - h0.T
# return np.sum(np.square(error))



def logSum(params,X,y):
    h0 = sigLin(params, X)
    cost = (-y*np.log(h0)) - ((1-y) * np.log(1-h0))
    return cost

def logSumgrad(params,X,y):
    h0 = sigLin(params, X)
    
    t1 = (-y*1/h0) * sigLinGrad(params, X)
    t2 =  ((1-y)*1/(1-h0)) * -sigLinGrad(params, X)
    grad = t1 - t2
    
    return grad

def testlogSum(params,X,y):
    h0 = sigLin(params, X)
    cost = - ((1-y) * np.log(1-h0))
    return cost

def testlogSumgrad(params,X,y):
    h0 = sigLin(params, X)
    
    t2 =  ((1-y)*1/(1-h0)) * -sigLinGrad(params, X)
    grad = - t2
    
    print grad
    return grad

X = np.asarray([3, 5])
params = np.asarray([2, 1])
y = 0

print check_grad(logSum, logSumgrad, params,X,y)

X = np.asarray([-1, -2])
params = np.asarray([2, 1])
y = 0

print check_grad(testlogSum, testlogSumgrad, params,X,y)

X = np.random.random([2,10])
params = np.random.random([10])
y = np.asarray([1,0])

X = np.asarray([[3, 5],[0,1],[-2,-5]])
params = np.asarray([2, 1])
y = np.asarray([1,0,0])

def logSumX(params,X,y):
    h0 = sigLin(params, X)
    cost = logLoss(y,h0)
    return cost

def logSumgradX(params,X,y):
    h0 = sigLin(params, X)
    h0gX = sigLinGrad(params, X)
    return logLossGrad(y, h0, h0gX)

print check_grad(logSumX, logSumgradX, params,X,y)

def sigSum(params,X,y):
    h0 = sigLin(params, X)
    return squaredLoss(y, h0)

def sigSumgrad(params,X,y):
    h0 = sigLin(params, X)
    h0gX = sigLinGrad(params, X)
    return squaredLossGrad(y, h0, h0gX)

def prod(params,X):
    
    return np.sum((np.dot(X,params)))

def prodGrad(params,X):
    
    return X

X = np.random.random([2,10])
params = np.random.random([10])
y = np.asarray([1,0])


print check_grad(sigSum, sigSumgrad, params,X,y)

def sigSq(X):
    
    return np.square(sigmoid(X))

def sigSqgrad(X):
    
    return 2*(sigmoid(X))*sigmoidGrad(X)

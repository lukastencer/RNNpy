from scipy.optimize import check_grad,approx_fprime,fmin,fmin_bfgs,fmin_cg
import numpy as np
from sigmoid import sigmoid,sigmoidGrad
from loss_functions import *
from hypothesis import *

DEBUG = False

class LogisticRegressionNet():

    def __init__(self, optimize = 'fmin', loss = squaredLoss):
        self.models = {}
        self.labels = None
        self.optimize = None
        self.loss = loss
        self.h0 = sigLin
        self.h0grad = sigLinGrad
        
        
        if optimize == 'fmin':
            self.optimize = fmin
        elif optimize == 'fmin_bfgs':
            self.optimize = fmin_bfgs

    def fit(self,X,y):
        
        self.labels = np.unique(y)
        
        for label in self.labels:
            
            yindi = np.zeros(y.shape)
            yindi[y==label] = 1
            
            self.models[label] = LogisticRegressionBin().fit(X, yindi)
    
    def predict(self,X):
        
        results = []
        
        for label in self.labels:
            tempRes = self.models[label].predict_proba(X)
            results.append(tempRes)
            
        return self.labels[np.argmax(np.asarray(results),0)]

class LogisticRegressionBin():
    
    def __init__(self, optimize = 'fmin',loss = logLoss, lossGrad = logLossGrad):
    
        self.params = None
        
        self.loss = loss
        self.lossGrad = lossGrad
        self.h0 = sigLin
        self.h0grad = sigLinGrad
        
        if optimize == 'fmin':
            self.optimize = fmin
            self.optimizeOrder = 0
        elif optimize == 'fmin_bfgs':
            self.optimize = fmin_bfgs
            self.optimizeOrder = 1
            
    
    def fit(self,X,y,initParams = None):
        self.params = np.zeros([1,X.shape[1]+1])
        self.labels = np.unique(y)
        X_nopad = X
        X = np.pad(X,((0,0),(1,0)),mode='constant',constant_values=1)
        
        #print self.cost(self.params,X, y)
        
        if initParams is None:
            init = np.random.random(self.params.size)
            #init = np.zeros(self.params.size)
        else:
            init = initParams
        
        if DEBUG:
            _epsilon = np.sqrt(np.finfo(float).eps)
            #print approx_fprime(self.params[0], self.cost, _epsilon, X,y)
            print check_grad(self.cost, self.grad, init,X,y)
        
        if self.optimizeOrder == 0:
            self.params = self.optimize(self.cost,init,args=(X,y),disp=False)
        if self.optimizeOrder == 1:
            self.params = self.optimize(self.cost,init,self.grad,args=(X,y),disp=False)
            
        return self
    
    def cost(self,params,X,y):
        if params is None:
            h0 = self.h0(self.params,X)
        else:
            h0 = self.h0(params,X)
        return self.loss(y,h0)
    
    def grad(self,params,X,y):
        if params is None:
            h0 = self.h0(self.params,X)
            h0Grad = self.h0grad(self.params,X)
        else:
            h0 = self.h0(params,X)
            h0Grad = self.h0grad(params,X)
        return self.lossGrad(y,h0,h0Grad)
    
    def predict(self,X):
        
        X = np.pad(X,((0,0),(1,0)),mode='constant',constant_values=1)
        
        h0 = sigmoid(np.dot(X,self.params.T))

        return h0 > 0.5
    
    def predict_proba(self,X):
        
        X = np.pad(X,((0,0),(1,0)),mode='constant',constant_values=1)
        
        h0 = sigmoid(np.dot(X,self.params.T))

        return h0
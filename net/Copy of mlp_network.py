import numpy as np
from activation_functions import *
from loss_functions import *
from scipy.optimize import fmin,fmin_bfgs,fmin_cg,check_grad

DEBUG = True

class MLP:
    
#     layersSize = None
#     layers = None
#     regularizations = None
#     loss = None
#     labels = None
#     yindi = None
#     
#     paramSplits = None
    
    def __init__(self,hiddenLayersSize=None,hiddenLayersType=None,output='sig',loss = 'sqr',regularizations=None, optimize = 'fmin'):
        """
            possible loss functions: sqr, TODO: abs, crossent, hinge, negloglik ; http://www.iro.umontreal.ca/~bengioy/ift6266/H12/html.old/mlp_en.html
        """
        self.optimize = None
        self.optimizeOrder = None
        
        if optimize == 'fmin':
            self.optimize = fmin
            self.optimizeOrder = 0
        elif optimize == 'fmin_bfgs':
            self.optimize = fmin_bfgs
            self.optimizeOrder = 1
        elif optimize == 'fmin_cg':
            self.optimize = fmin_cg
            self.optimizeOrder = 1
        
        if hiddenLayersSize is None:
            hiddenLayersSize = [10]
            
        if hiddenLayersType is None:
            hiddenLayersType = ['sig']
        
        self.layersSize = hiddenLayersSize
        self.regularizations = regularizations
        self.layers = []
        self.paramSplits = []
        self.labels = None
        self.yindi = None
        self.loss = None
        self.lossGrad = None
        
        hiddenLayersType.append(output)
        
        for htype in hiddenLayersType:
            if htype == 'sig':
                self.layers.append(Layer(activFunc=sigmoid))
                
        if loss == 'sqr':
            self.loss = squaredLoss
            self.lossGrad = squaredLossGrad
        
    def fit(self,X,y):
        
        inDim = X.shape[1]
        
        self.layersSize.append(len(np.unique(y)))
        self.layersSize.insert(0, inDim)
        self.setIndi(y)
        
        paramSum = 0
        for i,layer in enumerate(self.layers):
            layer.initParams([self.layersSize[i+1],self.layersSize[i]])
            split = self.layersSize[i+1] * self.layersSize[i]
            paramSum += split
            self.paramSplits.append(paramSum)
        
        init = self.getParams()
        
        if DEBUG:
            _epsilon = np.sqrt(np.finfo(float).eps)
            #print approx_fprime(self.params[0], self.cost, _epsilon, X,y)
            print check_grad(self.cost, self.grad, init,X,self.yindi)
        
        if self.optimizeOrder == 0:
            newParams = self.optimize(self.cost,init,args=(X,self.yindi),disp=False)
        if self.optimizeOrder == 1:
            newParams = self.optimize(self.cost,init,args=(X,self.yindi),disp=False)
        
#        newParams = self.optimize(self.cost, self.getParams(), args = (X,y))
        
        self.setParams(newParams)

    def setIndi(self,y):
        
        self.labels = np.unique(y)
        allIndi = []
        
        for label in self.labels:
            
            yindi = np.zeros(y.shape)
            yindi[y==label] = 1
            allIndi.append(yindi)
            
        self.yindi= np.asarray(allIndi).T
        
    def predict(self,X):
        return self.labels[np.argmax(self.forwardPass(X=X),axis=1)]
    
    def forwardPass(self,params=None,X=None):
        tempRes = X
        
        if params is None:
            for layer in self.layers:
                tempRes = layer.forward(inX=tempRes)
        else:
            for layer,lparam in zip(self.layers,params):
                tempRes = layer.forward(lparam,tempRes)
                            
        return tempRes
        
    def getParams(self):
        outParams = []
        for layer in self.layers:
            outParams.append(layer.getParams().flatten())
        return np.concatenate(outParams)

    def setParams(self,params):

        paramList = self.unfoldParams(params)        
        
        for param, layer in zip(paramList,self.layers):
            layer.setParams(param)
        
    def cost(self,params,X,y):
        
        paramList = self.unfoldParams(params)
            
        return np.sum(self.loss(y,self.forwardPass(paramList, X)))
    
    def grad(self,params,X,y):
        
        paramList = self.unfoldParams(params)
        
        predicted = self.forwardPass(paramList, X)
        delta = self.lossGrad(y,predicted) # should be substracted from original values
        
        allgrad = []
        
        for i,layer in enumerate(reversed(self.layers)):
            
#             if i == 0 :
#                 continue
            
            layerGrad = np.dot(layer.lastIncoming.T,delta).T
            
            allgrad.insert(0, layerGrad.flatten())
            
            delta = np.dot(layer.params * delta) * layer.grad(layer.lastPass) #TODO the last thing to do right!, should be n_samplesXnum_neurons
        
        return self.unfoldParams(allgrad)
    
    def unfoldParams(self,params):
        
        paramList = np.split(params,self.paramSplits[:-1])
        
        
        for i,elem in enumerate(paramList):
            shape = self.layers[i].params.shape
            #shape = [self.layersSize[i+1],self.layersSize[i]]
            paramList[i] = np.reshape(elem,shape)
            
        return paramList
    
    
class Layer():
    
#     params = None
#     regularizations = None
#     activation = None
    
    def __init__(self,activFunc,size=None):
        self.params = None
        self.regularizations = None
        self.activation = activFunc
        self.grad = eval(activFunc.__name__ + 'Grad')
        self.lastPass = None
        self.lastActivation = None
        self.lastIncoming = None
        
        if size is not None:
            self.params = np.random.random(size)
    
    def initParams(self,size):
        self.params = np.random.random(size)
    
    def forward(self,params=None,inX=None):
        
        if params is None: 
            dprod = np.dot(inX,self.params.T)
            self.lastPass = dprod
            self.lastActivation = self.activation(dprod)
            self.lastIncoming = inX
            return self.lastActivation
        else:
            dprod = np.dot(inX,params.T)
            self.lastPass = dprod
            self.lastActivation = self.activation(dprod)
            self.lastIncoming = inX
            return self.lastActivation
        
    def getParams(self):
        return self.params
    
    def getParamsFlat(self):
        return self.params.flatten()
        
    def setParams(self,params):
        self.params = params
        
    def setParamsFlat(self,params,size):
        self.params = np.reshape(params,self.params.shape)
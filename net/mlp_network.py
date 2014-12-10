import numpy as np
from activation_functions import *
from loss_functions import *
from scipy.optimize import fmin,fmin_bfgs,fmin_cg,check_grad

DEBUG = False

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
            hiddenLayersSize = [3]
            hiddenLayersSize = []
            
        if hiddenLayersType is None:
            hiddenLayersType = ['sig']
            hiddenLayersType = []
        
        self.layersSize = hiddenLayersSize
        self.regularizations = regularizations
        self.layers = []
        self.paramSplits = []
        self.labels = None
        self.yindi = None
        self.loss = None
        self.lossGrad = None
        
        hiddenLayersType.append(output)
        
        self.layers.append(Layer(activFunc=inputLayer))
        
        for htype in hiddenLayersType:
            if htype == 'sig':
                self.layers.append(Layer(activFunc=sigmoid))
                
        if loss == 'sqr':
            self.loss = squaredLoss
            self.lossGrad = squaredLossGrad
        
    def fit(self,X,y,initParams = None):
        
        X = np.pad(X,((0,0),(1,0)),mode='constant',constant_values=1)
        
        inDim = X.shape[1]
        
#         if DEBUG:
# #             self.layersSize.append(1)
# #             self.layersSize.insert(0, int(inDim))
# #             self.yindi = np.asarray(np.logical_not(y),dtype=np.int32)         
#         else:            
        self.layersSize.append(len(np.unique(y)))
        self.layersSize.insert(0, int(inDim))
        self.setIndi(y)
        
#         self.layersSize[-1]=1
#         self.yindi = np.expand_dims(self.yindi[:,0].T,1)
        
        paramSum = 0
        for i,layer in enumerate(self.layers):
            if not( i == len(self.layers)-1):
                layer.initParams([self.layersSize[i+1],self.layersSize[i]])
                split = self.layersSize[i+1] * self.layersSize[i]
                paramSum += split
                self.paramSplits.append(paramSum)
            else:
                layer.setParams(None)

        if initParams is None:
            init = self.getParams()
        else:
            init = initParams

        
        if DEBUG:
            _epsilon = np.sqrt(np.finfo(float).eps)
            #print approx_fprime(self.params[0], self.cost, _epsilon, X,y)
            print check_grad(self.cost, self.grad, np.zeros(init.shape),X,self.yindi)
            print check_grad(self.cost, self.grad, init,X,self.yindi)
        
        if self.optimizeOrder == 0:
            newParams = self.optimize(self.cost,init,args=(X,self.yindi),disp=False)
        if self.optimizeOrder == 1:
            newParams = self.optimize(self.cost,init,args=(X,self.yindi),disp=False)
        
        #newParams = self.optimize(self.cost, self.getParams(), args = (X,y))
        
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
        X = np.pad(X,((0,0),(1,0)),mode='constant',constant_values=1)
        
        return self.labels[np.argmax(self.forwardPass(X=X),axis=1)]
    
    def forwardPass(self,params=None,X=None):
        tempRes = X
        
        
        if params is None:
            #params += [[]]*(len(self.layers)-len(self.params))
            for layer in self.layers:
                tempRes = layer.forward(inX=tempRes)
        else:
            params += [[]]*(len(self.layers)-len(params))
            for layer,lparam in zip(self.layers,params):
                tempRes = layer.forward(lparam,tempRes)
                            
        return tempRes
        
    def getParams(self):
        outParams = []
        for layer in self.layers:
            params = layer.getParams()
            if params is not None:
                outParams.append(params.flatten())
        return np.concatenate(outParams)

    def setParams(self,params):

        paramList = self.unfoldParams(params)        
        
        for param, layer in zip(paramList,self.layers):
            layer.setParams(param)
        
    def cost(self,params,X,y):
        
        paramList = self.unfoldParams(params)
        
        fpass = self.forwardPass(paramList, X)
        
#         print y
#         print fpass
#         print np.sum(self.loss(y,fpass))
        return self.loss(y,fpass)
    
    def grad(self,inParams,X,y):
        
        paramList = self.unfoldParams(inParams)
        
        paramList += [np.array([])]*(len(self.layers)-len(paramList))
        
        predicted = self.forwardPass(paramList, X)
        delta = self.lossGrad(y,predicted) # should be substracted from original values
        
        allgrad = []
        
        for i,(layer,params) in enumerate(zip(reversed(self.layers),reversed(paramList))):
            
#             if i == 0 :
#                 continue
            if layer.params is not None:
                layerGrad = np.dot(layer.lastIncoming.T,delta).T
                #print layerGrad
                
                allgrad.insert(0, layerGrad.flatten())
                
                if params.size == 0:
                    delta = np.dot(layer.params.T, delta.T).T * layer.grad(layer.lastIncoming) #TODO the last thing to do right!, should be n_samplesXnum_neurons
                else:
                    delta = np.dot(params.T, delta.T).T * layer.grad(layer.lastIncoming)
        
        return np.concatenate(allgrad)
    
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
        self.lastProd = None
        self.lastActivation = None
        self.lastIncoming = None
        
        if size is not None:
            self.params = np.random.random(size)
    
    def initParams(self,size,initFunc = 'rand'):
        
        if initFunc == 'zero':
            self.params = np.zeros(size)
        elif initFunc == 'rand':
            self.params = np.random.random(size)
    
    def forward(self,params=None,inX=None):
        if self.params is not None:
            if params is None: 
                self.lastIncoming = inX
                self.lastActivation = self.activation(inX)
                self.lastProd = np.dot(inX,self.params.T)
                return self.lastProd
            else:
                self.lastIncoming = inX
                self.lastActivation = self.activation(inX)
                self.lastProd = np.dot(inX,params.T)
                return self.lastProd
        else:
                self.lastActivation = self.activation(inX)
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
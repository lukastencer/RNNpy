from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris,make_hastie_10_2,make_classification,make_blobs
from sklearn.linear_model import LogisticRegression
from net import LogisticRegressionNet,MLP,LogisticRegressionBin
import numpy as np

data = load_iris()
inX = data.data
iny = data.target

data = make_classification(n_features=4,n_classes=2)
inX = data[0]
iny = data[1]
 
data = make_blobs(n_features=2,centers=2,n_samples=10)
inX = data[0]
iny = data[1]

for _ in xrange(50):

    X, tX, y, ty = train_test_split(inX,iny,test_size=0.33)
    
    clf1 = LogisticRegression()
    
    clf2 = LogisticRegressionNet(optimize='fmin_bfgs')
    #clf2 = LogisticRegressionBin(optimize='fmin_bfgs')
    
    clf3 = MLP(optimize='fmin_bfgs')
    
#     X = [[0],[2]]
#     y = [0,1]
#     params = [0,1]
#      
#     X = np.asarray(X)
#     y = np.asarray(y)
#     params = np.asarray(params)
    
    
    clf1.fit(X, y)
    
#     X = np.asarray([[3, 5],[3,5],[3,5]])
#     params = np.asarray([2, 1])
#     y = np.asarray([1,1,1])
    
    clf2.fit(X, y)
    clf3.fit(X, y)
    
    res1 = clf1.predict(tX)
    res2 = clf2.predict(tX)
    res3 = clf3.predict(tX)
    
    print accuracy_score(ty, res1)
    print accuracy_score(ty, res2)
    print accuracy_score(ty, res3)
    print '------'
    
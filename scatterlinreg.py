import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(self):
        pass
       

    def trainingset(self,X,Y):  
        res=inv(X.T.dot(X)).dot(X.T).dot(Y)
        return res

    def testingset(self,X,param):
        ans=X.dot(param)
        return ans

lr=LinearRegression()   
data=np.array([[4,33],[4.5,42],[5,45],[5.5,51],[6,53],[6.5,61],[7,62]])

X=data[:,0]
y=data[:,1]


print (X)
print (y)
xt=np.transpose(X)

X=X.reshape(len(X),1)

#print (X)
res=lr.trainingset(X,y)
print(res)
yp=lr.testingset(X,res)
print (yp)
plt.scatter(X,y)
plt.plot(X,yp,"b")
plt.show()
print("__")
print(sum(X))
print(sum(yp))
sumofyp=sum(yp)
ysum=sum(y)
ypmean=np.mean(yp)
print(ypmean)
print(ysum)
#print(sumofyp-ypmean)
#print(sum(y)-ypmean)
ymean=np.mean(y)
print(pow(sum(yp-ymean),2))



import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("data.csv")
df.describe()
dfTrain=df.head(int(len(df)*2/3))
dfTest=df.tail(int(len(df)/3))
# print(dfTrain)
# print(dfTest)
corr=dfTrain.corr()
print(corr)
ind=['sqft_lot','floors','bedrooms','sqft_above','sqft_basement','waterfront','view','yr_built','yr_renovated','condition']

x=[]
for i in range(len(dfTrain)):
    temp=[]
    temp.append(1)
    for j in ind:
        temp.append(dfTrain[j][i])
    x.append(temp)
Y=dfTrain['price']
Y=np.array(Y)

X=np.array(x)
Xt=np.transpose(X)
XtX=Xt.dot(X)
XtXi=np.linalg.inv(XtX)
XtY=Xt.dot(Y)
W=XtXi.dot(XtY)


for i in range(len(W)):
    print("w"+str(i)+" : "+str(W[i]))

Ypred = []
for i in range(len(dfTrain)):
    sum=W[0]
    for j in range(len(ind)):
        sum+=W[j+1]*dfTrain[ind[j]][i]
    Ypred.append(sum)

SSE=0
Y_=0
for i in range(len(Y)):
    SSE+=(Y[i]-Ypred[i])**2
    Y_+=Y[i]
Y_/=len(Y)
SST=0
for i in range(len(Y)):
    SST+=(Y[i]-Y_)**2
SSR=SST-SSE
Rsq=SSR/SST
print("SSR : ",SSR)
print("SSE : ",SSE)
print("SST : ",SST)
print("R^2: "+str(Rsq))
adjRsq=1-((1-Rsq)*(len(X)-1)/(len(X)-len(ind)-1))
print("Adjusted R^2: "+str(adjRsq))

x1=[]
for i in range(len(X)):
    x1.append(x[i][1])
print(plt.scatter(x1,Y))

price=[]
for i in range(len(dfTest)):
    sum=W[0]
    for j in range(len(ind)):
        sum+=W[j+1]*dfTest[ind[j]][i+3067]
    price.append(sum)

Y=[]
for i in range(len(dfTest)):
    Y.append(dfTest['price'][i+3067])
    
for i in range(len(Y)):
    print(str(Y[i])+"\t"+str(price[i]))

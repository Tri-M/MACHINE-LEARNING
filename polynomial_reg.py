import pandas as pd
import numpy as np
import sys

df = pd.read_csv('iris.csv')

def calcB(X, Y, degree):
    xmat = []
    for i in range(degree+1):
        lis = []
        for j in X:
            if i==0:
                lis.append(1)
            else:
                lis.append(j**i)
        xmat.append(lis)
    xT = np.array(xmat)
    y = np.array(Y)
    x = xT.transpose()
    xTx = np.dot(xT, x)
    xTxI = np.linalg.inv(xTx)
    xTy = np.dot(xT, y)
    B = np.dot(xTxI, xTy)
    
    return B

def calcMetrics(B, X, Y):
    result = []
    Ymean = np.mean(np.array(Y))
    
    SSE, SSR, SST = 0, 0, 0
    
    for j in range(len(X)):
        Yhat = 0
        for i in range(len(B)):
            if i==0:
                Yhat += B[i]
            else:
                Yhat += B[i]*X[j]**i
        SSE += (Yhat-Y[j])**2
        SST += (Y[j]-Ymean)**2
        SSR += (Yhat-Ymean)**2
    result.append(SSE)
    result.append(SSR)
    result.append(SST)
    result.append(SSR/SST)
    result.append((SSR/SST)**0.5)
    return result
            
def display(X, Y):
    result = []
    for i in range(1, 4):
        B = calcB(X, Y, i)    
        result.append(calcMetrics(B, X, Y))
        print("\nDegree", i)
        print("SSE: ", result[i-1][0])
        print("SSR: ", result[i-1][1])
        print("SST: ", result[i-1][2])
        print("r^2: ", result[i-1][3])
        print("r: ", result[i-1][4])
    return result

petalLength = df['petallength']
petalWidth = df['petalwidth']
sepalLength = df['sepallength']
sepalWidth = df['sepalwidth']

SSE = []
degree = 2

print("\nPetal Length Vs Petal Width")
result = display(petalLength, petalWidth)
SSE.append(result[degree-1][0])

print("\nPetal Length Vs Sepal Length")
result = display(petalLength, sepalLength)
SSE.append(result[degree-1][0])

print("\nPetal Length Vs Sepal Width")
result = display(petalLength, sepalWidth)
SSE.append(result[degree-1][0])

print("\nSepal Length Vs Sepal Width")
result = display(sepalLength, sepalWidth)
SSE.append(result[degree-1][0])

print("\nPetal Width Vs Sepal Width")
result = display(petalWidth, sepalWidth)
SSE.append(result[degree-1][0])

print("\nPetal Width Vs Sepal Length")
result = display(petalWidth, sepalLength)
SSE.append(result[degree-1][0])

print("\nMinimum SSE: ", min(SSE), SSE.index(min(SSE)))

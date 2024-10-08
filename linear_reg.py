# import numpy as np   
# import matplotlib.pyplot as plt
# import pandas as pd

# df = pd.read_csv (r'data.csv')
# fig, ax = plt.subplots(3,3,figsize=(10,7))
# bedr=df['bedrooms']
# bathr=df['bathrooms']
# sqft=df['sqft_living']
# price=df['price']
# flr=df['floors']
# sqft_abv=df['sqft_above']
# sqft_bse=df['sqft_basement']
# view=df['view']
# cond=df['condition']
# waterfront=df['waterfront']

# ax[0,0].scatter(bedr,price,color='c')
# ax[0, 0].set_title("bedroom v/s price")


# ax[0,1].scatter(sqft_abv,price,color='y')
# ax[0, 1].set_title("sqft_above v/s price")


# ax[0,2].scatter(sqft,price,color='r')
# ax[0,2].set_title("sqft v/s price")


# ax[1,0].scatter(flr,price,color='g')
# ax[1,0].set_title("floors v/s price")


# ax[1,1].scatter(sqft_bse,price,color='m')
# ax[1,1].set_title('basement sqft v/s price')

# ax[1,2].scatter(bathr,price,color='b')
# ax[1,2].set_title("Bathroom v/s price")

# ax[2,0].scatter(view,price,color='c')
# ax[2,0].set_title("view v/s price")

# ax[2,1].scatter(cond,price,color='y')
# ax[2,1].set_title("condition v/s price")

# ax[2,2].scatter(waterfront,price,color='b')
# ax[2,2].set_title("waterfront v/s price")

# fig.tight_layout() 
# plt.show()




# def LinReg(ar):
    
#     X=[]
#     for i in ar:
#         X.append([1,i])
# # print(sqftX)

#     print("X transpose")
#     X_T=np.transpose(X)
#     print(X_T)
    
#     print("\n X^T X :")
#     XT_X=np.dot(X_T,X)
#     print(XT_X)
#     XtXi=np.linalg.inv(XT_X)
#     print("\n x transpose x inverse")
#     print(XtXi)
    
#     Y=[]
#     for i in price:
#         Y.append(int(i))
#     # print(Y)
    
#     print(" x transpose Y")
#     XtY=np.dot(X_T,Y)
#     print(XtY)
    
#     print("\n B matrix B0 B1")
#     B=np.dot(XtXi,XtY)
#     print(B)
    
#     print("\nLinear equation ")
#     if(B[0]<0):
#         print("y="+str(B[1])+"x"+str(B[0]))
#     else:
#         print("y="+str(B[1])+"x+"+str(B[0]))
    
    
    
#     x = np.linspace(0,len(ar))
#     y = B[1]*x+B[0]
#     plt.plot(x, y, '--', label='y=2x+1',color='g')
#     plt.show()
    
#     SSE=0
#     SE=0
#     for i in range(len(price)):
#         SE+=sqft[i]-(B[1]*sqft[i]+B[0])
#         SSE+=(price[i]-(B[1]*ar[i]+B[0]))**2   #sum of squared error
#     print("Errors sum: "+str(SE)+"\nSSE: "+str(SSE))
    
#     Y_ = np.mean(Y)
#     print("\nMean (Y_): "+str(Y_))
    
#     S=0
#     for i in price:
#         S+=i-Y_
#     print("Sum of Yi-Y_: "+str(S))
    
#     SST=0
#     for i in ar:
#         SST+=(i-Y_)**2
#     print("SST: "+str(SST))
    
#     SSR=SST-SSE
#     print("SSR: "+str(SSR))


#     R2=SSR/SST
#     print("R2: "+str(R2))
# print("\n")
# print("\nLinear regression using sqft")
# print("\n")
# LinReg(sqft)
# print("________________________________________________")
# print("\nLinear Regression using bathrooms ")
# print("\n")
# LinReg(bathr)
# print("________________________________________________")
# print("\nLinear Regression using Bedrooms ")
# print("\n")
# LinReg(bedr)
# print("________________________________________________")
# print("\nLinear Regression using floors ")
# print("\n")
# LinReg(flr)
# print("________________________________________________")

import numpy as np
import pandas as pd
import matplotlib.pyplot as pt


def linearRegression(x, y):
    X = np.array([[1, i] for i in x])
    Y = np.array([[i] for i in y])
    B = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose().dot(Y))
    print(B)
    b0 = B[0][0]
    b1 = B[1][0]

    return (b0, b1)


def calcSSE(x, y):
    b0, b1 = linearRegression(x, y)
    SSE = 0
    n = len(x)
    for i in range(n):
        SSE += (y[i]-(b0+b1*x[i]))**2
    return SSE


def calcSSR(x, y):
    b0, b1 = linearRegression(x, y)
    ybar = sum(y)/len(y)
    SSR = 0
    n = len(x)
    for i in range(n):
        SSR += ((b0+b1*x[i])-ybar)**2
    return SSR


def evaluation(x, y):
    sse = calcSSE(x, y)
    ssr = calcSSR(x, y)
    r2 = ssr/(ssr+sse)
    return (ssr, sse, r2)


df = pd.read_csv('data.csv')
features = list(df.columns)
print(features)
Y = 'price'
X = features
X.remove('price')
print(X)

# for i in X:
#       df.plot.scatter(x=i,y=Y,s=2)

X1 = X[1:-4]
print(X1)
for i in X1:
    x = df[i].tolist()
    y = df[Y].tolist()
    b0, b1 = linearRegression(x, y)
    print(i+"vs"+Y)
    print("\nThe linear Regression is\n")
    print("y="+str(b1)+"*x+"+str(b0)+"\n")

    ssr, sse, r2 = evaluation(x, y)
    print("SSR:", ssr)
    print("SSE:", sse)
    print("Coefficient of Determination:", r2, "\n")

    pt.scatter(x, y)
    ub = max(x)
    x = np.linspace(0, ub, 1000)
    y = b0+b1*x
    pt.plot(x, y, color="orange")
    pt.xlabel(i)
    pt.ylabel(Y)
    pt.show()

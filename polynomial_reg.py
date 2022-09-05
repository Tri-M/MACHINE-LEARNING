import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def polynomial_regression(X, Y, d):
    B = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose().dot(Y))
    return B

def preparePlot(x, y, B, title=""):
    plt.scatter(x, y)
    
    X = np.array([[i**p for p in range(d+1)] for i in np.linspace(0, 10, 100)])
    
    Ycap = X.dot(B)
    
    plt.plot(np.linspace(0, 10, 100), Ycap, color="red")
    plt.xlabel(cols[i])
    plt.ylabel(cols[j])
    plt.title(title)
    plt.show()
    
def calcSSE(X, Y, B):
    n = len(Y)

    Ycap = X.dot(B)
    error = Y - Ycap
    
    sse = 0
    for i in range(n):
        sse += error[i][0]**2
    return sse

def calcSSR(X, Y, B):
    n = len(Y)
    
    ybar = np.mean(Y)
    Ycap = X.dot(B)
    
    ssr=0
    for i in range(n):
        ssr += (Ycap[i][0]-ybar)**2
    return ssr


def evaluate(X, Y, B):
    sse = calcSSE(X, Y, B)
    ssr = calcSSR(X, Y, B)
    r2 = ssr/(ssr+sse)
    return (sse, ssr, r2)


#execution starts here
data = pd.read_csv("iris.csv")
cols = list(data.columns)
print(cols)


max_r2= 0
optimum_model = ""

for d in range(1, 4):
    print("------------------------------\nd=", d)
    for i in range(0, 4):
        for j in range(i+1, 4):
            
            title = cols[i]+ "  VS  "+ cols[j] + " with d=" + str(d)
            print("\n\n", title)
            
            x = data[cols[i]].tolist()
            y = data[cols[j]].tolist()
            
            X = np.array([[i**p for p in range(d+1)] for i in x])
            Y = np.array([[i] for i in y])
            
            B = polynomial_regression(X, Y, d)
            print(B)
            sse, ssr, r2 = evaluate(X, Y, B)
            
            print("[SSE]: ", sse)
            print("[SSR]: ", ssr)
            print("[SST]: ", sse + ssr)
            print("[r^2]: ", r2)
            
            
            
            
            if r2> max_r2:
                max_r2 = r2
                optimum_model = title  
            
            preparePlot(x, y, B, title)
            
            
print("---------------------------------\n[The optimum model]: ", optimum_model)




# import pandas as py
# import numpy as np
# import matplotlib.pyplot as plt


# def poly(X, Y, d):
#     plt.scatter(X, Y)
#     # plt.show()
#     n = len(X)
#     matrix = np.array([[i ** p for p in range(d + 1)] for i in X])
#     y = np.array([[i] for i in Y])

#     xt = np.transpose(matrix)
#     xm = (np.dot(xt, matrix))
#     # print(xm)
#     xi = (np.linalg.inv(xm))
#     # print(xi)
#     ym = (np.dot(xt, y))
#     # print(ym)
#     b = np.dot(xi, ym)
#     print(b)
#     y1 = matrix.dot(b)
#     sse = 0
#     for i in range(0, len(y)):
#         sse += (y[i] - y1[i]) ** 2
#         # se+=y[i]-y1[i]
#     print("sum of squared error", sse)
#     mean = sum(y) / len(y)
#     sst = 0
#     for i in range(0, len(y)):
#         sst += (y[i] - (mean)) ** 2
#     print("total sum of square", sst)
#     ssr = 0
#     for i in range(len(y)):
#         ssr += (y1[i] - mean) ** 2
#     print("Regression sum of squares", ssr)
#     print("coefficient of determination", ssr / sst)
#     print(len(y1))
#     X = np.array([[i ** p for p in range(d + 1)] for i in np.linspace(0, 10, 100)])

#     Ycap = X.dot(b)
#     plt.plot(np.linspace(0, 10, 100), Ycap, color="red")
#     return (b)


# data = py.read_csv("iris.csv")
# sl = data["SepalLengthCm"]
# sw = data["SepalWidthCm"]s
# pl = data["PetalLengthCm"]
# pw = data["PetalWidthCm"]
# print("\nSepal length vs sepal width")
# co = poly(list(sl), list(sw), 2)
# plt.show()

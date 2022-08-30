
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def multiple_regression(X, Y):
    W = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose().dot(Y))
    return W

def outliers_apply(df, feature):
    q1, q3 = np.percentile(df[feature], [25, 75])
    IQR = q3 - q1 
    lower_range = q1 - (3 * IQR) 
    upper_range = q3 + (3 * IQR)
    to_drop = df[(df[feature]<lower_range)|(df[feature]>upper_range)]
    df.drop(to_drop.index, inplace=True)
    
def calcSSE(X, Y, W):
    n = len(Y)

    Ycap = X.dot(W)
    error = Y - Ycap
    
    sse = 0
    for i in range(n):
        sse += error[i][0]**2
    return sse

def calcSSR(X, Y, W):
    n = len(Y)
    
    ybar = np.mean(Y)
    Ycap = X.dot(W)
    
    ssr=0
    for i in range(n):
        ssr += (Ycap[i][0]-ybar)**2
    return ssr

def evaluate(X, Y, W):
    sse = calcSSE(X, Y, W)
    ssr = calcSSR(X, Y, W)
    r2 = ssr/(ssr+sse)
    _r2 = 1 - (1-r2)*(len(X)-1)/(len(X)-len(W))
    return (sse, ssr, r2, _r2)


df = pd.read_csv("data.csv")
cols = list(df.columns)
print(cols)

features = ["bedrooms", "bathrooms",  "sqft_living", "sqft_lot", "sqft_above", "sqft_basement"]
print("\n\nFeatures considered:\n", features)

final_features = [x for x in features]
p = df[features].corr().values.tolist()
for i in range(len(p)):
    for j in range(i+1, len(p)):
        if abs(p[i][j]) > 0.7 and features[i] in final_features:
            final_features.remove(features[i])
print("\n\nFeatures after removing multicollinearity:\n", final_features)


print("\n\nSize of dataset before and after removing outliers:\n", len(df), end=" ")
for i in features:
    plt.scatter(df[i], df["price"])
    plt.show()

outliers_apply(df, "price")

print(len(df))
for i in features:
    plt.scatter(df[i], df["price"])
    plt.show()



train, test = train_test_split(df, test_size=0.2)
print("\n\nSize of training and testing set:\n", len(train), len(test))

X = np.array([[1]+[train[f].tolist()[i] for f in final_features] for i in range(len(train))])
Y = np.array([[i] for i in train["price"].tolist()])

W = multiple_regression(X, Y)
print("\n\nW:\n", W)

_X = np.array([[1]+[test[f].tolist()[i] for f in final_features] for i in range(len(test))])
_Y = np.array([[i] for i in test["price"].tolist()])

sse, ssr, r2, _r2 = evaluate(_X, _Y, W)

print("\n\nEvaluating the model on testing set:")
print("[SSE]: ", sse)
print("[SSR]: ", ssr)
print("[SST]: ", sse + ssr)
print("[r^2]: ", r2)
print("[adjusted r^2]: ", _r2)


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

dataset =pd.read_csv("bc.csv")
# print(dataset)
print(dataset.head())

x=list(dataset.columns)
x=x[2:]

final_features = [i for i in x]
p = dataset[x].corr().values.tolist()
for i in range(len(p)):
    for j in range(i+1, len(p)):
        if abs(p[i][j]) > 0.7 and x[i] in final_features:
            final_features.remove(x[i])
print("\n\nFeatures after removing multicollinearity:\n", final_features)
print(len(final_features))


X=[]
for i in range(len(dataset)):
    temp=[1]
    for j in range(1,len(final_features)):
        temp.append(dataset[final_features[j]][i])
    X.append(temp)
X=np.array(X)
y=dataset['diagnosis']
Y=[]
for i in range(len(y)):
    if y[i]=='M':
        Y.append(1)
    else:
        Y.append(0)
Y=np.array(Y)
# print(X)

# X=np.array(x)
Xt=np.transpose(X)
XtX=Xt.dot(X)
XtXi=np.linalg.inv(XtX)
XtY=Xt.dot(Y)
W=XtXi.dot(XtY)

X=[]
for i in range(len(dataset)):
    temp=[1]
    for j in range(1,len(final_features)):
        temp.append(dataset[final_features[j]][i])
    X.append(temp)
X=np.array(X)
y=dataset['diagnosis']
Y=[]
for i in range(len(y)):
    if y[i]=='M':
        Y.append(1)
    else:
        Y.append(0)
Y=np.array(Y)
# print(Y)

wtx=[0]*len(Y)

import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

THRESHOLD = 0.7

def calc(X, Y, W, theta):
    TP, TN, FP, FN = 0, 0, 0, 0
    Yhat = np.dot(X, W)
    N = len(X)
    for i in range(N):
        Ycap = 1 / (1 + math.exp(-Yhat[i]))

        if(Ycap >= theta):
            Ycap = 1
        else:
            Ycap = 0

        if(Ycap == 1 and Y[i] == 1):
            TP += 1
        elif(Ycap == 0 and Y[i] == 1):
            FN += 1
        elif(Ycap == 1 and Y[i] == 0):
            FP += 1
        else:
            TN += 1
           
    return [TP, FN, FP, TN]



df = pd.read_csv('data.csv')

cols = list(df.columns)[2:32]

features = list(cols)

corr = df[cols].corr().values

for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        if(abs(corr[i][j]) >= THRESHOLD and cols[i] in features and cols[j] in features):
            features.remove(cols[i])
7
# print(len(cols), len(features))
print(features)

train, test = train_test_split(df, test_size = 0.2)

trainSize = len(train)
train = train.reindex([i for i in range(0, trainSize)])

testSize = len(test)
test = test.reindex([i for i in range(0, testSize)])

print(trainSize, testSize)

# 1 - Malignant, 0 - Benign

X, Y = [], []

for i in range(trainSize):
    lis = [1]
    for j in features:
        lis.append(train[j][i])
    X.append(lis)
    if train['diagnosis'][i] == "M":
        Y.append(1)
    else:
        Y.append(0)

featureSize = len(features)
     
Wold, Wnew = [0 for i in range(featureSize+1)], [0 for i in range(featureSize+1)]
Wold, Wnew = np.array(Wold), np.array(Wnew)
X, Y = np.array(X), np.array(Y)
gradient = [0 for _ in range(len(X))]
eeta = 0.007

# iterations = 1000000

# while(True):
#     iterations-=1
#     for i in range(trainSize):
#         Yhat = X.dot(Wold)
#         sigmoid = 1 / (1 + math.exp(-1*Yhat[i]))
#         if(sigmoid >= 0.5):
#             Ycap = 1
#         else:
#             Ycap = 0
#         if(Ycap != Y[i]):
#             gradient = (Y[i] - sigmoid) * sigmoid * (1 - sigmoid) * X[i]
#             Wnew = Wold + eeta*gradient
#             Wold = Wnew

#     print(Wold)
#     if (np.all(gradient) == 0 or iterations == 0):
#             break
 
Wold = [ -1.56321027, -7.41984046, 0.19201655, 0.80786062, 4.56161812, -3.86185968,
         -1.37173668, 0.03099825, 0.20293206, 16.14072523, 2.4543871, -12.90958941 ]

# Wold = [-14.13967032, 7.95389434, -1.09796794, 5.13651019, -1.44968948, 14.10065082,
#         -2.88448603, 0.28892423, -5.05859146, 79.52821553, 0.547714, -46.33737769]

print("Coefficients")      
print(Wold)


XTest, YTest = [], []
for i in range(testSize):
    lis = [1]
    for j in features:
        lis.append(test[j][i])
    XTest.append(lis)
    if test['diagnosis'][i] == "M":
        YTest.append(1)
    else:
        YTest.append(0)

TP, FN, FP, TN = calc(XTest, YTest, Wold, 0.5)

print(TP, FN, FP, TN)

P = TP/(TP+FP)
R = TP/(TP+FN)

print("Accuracy: ", (TP+TN)/testSize)

print("Precision: ", P)

print("Recall: ", R)

print("F measure: ", 2*P*R/(P+R))

print("TPR: ", TP/(TP + FN))

print("FPR: ", FP/(FP + TN))


theta = np.linspace(0, 1, 10000)

TPR, FPR = [], []

for i in theta:
    TP, FN, FP, TN = calc(XTest, YTest, Wold, i)
    TPR.append(TP/(TP + FN))
    FPR.append(FP/(FP + TN))

FPR.extend([0, 1])
TPR.extend([0, 1])

FPR, TPR = zip(*sorted(zip(FPR, TPR)))

AUC = 0

for i in range(len(FPR)-1):
    AUC += 0.5*(FPR[i+1]-FPR[i])*(TPR[i]+TPR[i+1])
   
print("Area under the curve: ", AUC)

# print("Area under the curve: ", np.trapz(TPR, FPR))

plt.plot(FPR, TPR)
plt.show()
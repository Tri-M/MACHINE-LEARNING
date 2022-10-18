import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt



df = pd.read_csv('tic-tac-toe-endgame.csv')

one_hot_encoded_data = pd.get_dummies(df, columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9'])

features = list(one_hot_encoded_data.columns)[1:]

train, test = train_test_split(one_hot_encoded_data, test_size = 0.2)

trainSize = len(train)
testSize = len(test)

X = train[features]

Y = train['V10']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, Y)

plt.figure(figsize=(50, 30))

tree.plot_tree(dtree, feature_names=features)

plt.savefig("output.png")

plt.show()

prediction = dtree.predict(X)

Yarr = np.array(Y)

trainingError = 0

for i in range(trainSize):
    if prediction[i] != Yarr[i]:
        trainingError += 1

print("Training error: ", trainingError)

Ypred = dtree.predict(test[features])

testError = 0

Ytest = np.array(test['V10'])

for i in range(testSize):
    if Ypred[i] != Ytest[i]:
        testError += 1

print("Testing Error: ", testError)

print("Height: ", dtree.get_depth()+1)

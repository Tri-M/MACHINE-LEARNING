import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv("tic-tac-toe-endgame.csv")
print(df)
labels= preprocessing.LabelEncoder()                     
df=df.apply(labels.fit_transform) 
print(df)
feature_cols = ['V1','V2','V3','V4','V5','V6','V7','V8','V9']
X = df[feature_cols]                               
y = df.V10
print("X :",X)
print("y:" ,y)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
classifier =DecisionTreeClassifier(criterion="entropy", random_state=100)     
classf_gini=DecisionTreeClassifier(criterion="gini",random_state=100)
classf=classf_gini.fit(X_train,Y_train)

classifier.fit(X, y)    


y_pred= classifier.predict(X)  
y_pred1=classf_gini.predict(X)
print(y_pred)
print("Accuracy score :",accuracy_score(y,y_pred))
print("Gini index (accuracy) : ",accuracy_score(y,y_pred1))
print("confusion matrix :",confusion_matrix(y,y_pred))

    
    



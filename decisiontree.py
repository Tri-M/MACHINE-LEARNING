import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus

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
# # X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
classifier =DecisionTreeClassifier(criterion="entropy", random_state=100)     
classifier.fit(X, y)    
y_pred= classifier.predict(X)  
print(y_pred)


dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,filled=True, rounded=True,special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('tictactoe.png')
Image(graph.create_png())



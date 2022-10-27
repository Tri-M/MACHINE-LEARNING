import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

df=pd.read_csv("bc.csv")
print(df)
print(df.columns)
df["diagnosis"]=df["diagnosis"].map({'B':0,'M':1}).astype(int)
print(df.head())
corr=df.corr()
corr.nlargest(30,'diagnosis')['diagnosis']
x=df[['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean','radius_se','perimeter_se', 'area_se','compactness_se', 'concave points_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','texture_worst','area_worst']]
y=df[['diagnosis']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
predict = model.predict(x_test)
accuracy_score=accuracy_score(predict,y_test)
print(accuracy_score)
accuracy=model.score(x_train,y_train)
print(accuracy)



score=cross_val_score(model,x,y,cv=2)
print(score)

model=RandomForestClassifier(max_depth=6,random_state=5)
model.fit(x_train,y_train)
predict=model.predict(x_test)
acc = model.score(x_test,y_test)
print(acc)

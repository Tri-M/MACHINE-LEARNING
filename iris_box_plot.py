
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('iris.csv')

data.boxplot(column='sepallength', by='class')
data.boxplot(column='sepalwidth', by='class')
data.boxplot(column='petallength', by='class')
data.boxplot(column='petalwidth', by='class')



iris_setosa = data.loc[data["class"] == "Iris-setosa"]
iris_versicolor = data.loc[data["class"] == "Iris-versicolor"]
iris_virginica = data.loc[data["class"] == "Iris-virginica"]
dataset = [iris_setosa,iris_versicolor,iris_virginica]
for data in dataset:
    for i in data:
        if i!= 'class':
            print(i)
            #print("___________")
            Data1 = data[i]
          
            print("Minimum : ",Data1.min())
            print("maximum : ",Data1.max())
            print("1st quantile : ",Data1.quantile(0.25))
            print("2nd quantile: ",Data1.quantile(0.5))
            print("3rd quantile : ",Data1.quantile(0.75))
            print("___________")
            
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('heights.csv')


new_data = data[["Heights"]]
plt.figure(figsize = (10, 7))
new_data.boxplot()

print("Min height : ",data.Heights.min())
print("max height : ",data.Heights.max())
print("Q1 , 1st quartile: ",data.Heights.quantile(0.25))
print("Q2, 2nd quartile: ",data.Heights.quantile(0.5))
print("Q3: 3rd quartile: ",data.Heights.quantile(0.75))
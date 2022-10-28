import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, normalize

# 1. Sample code number: id number
# 2. Clump Thickness: 1 - 10
# 3. Uniformity of Cell Size: 1 - 10
# 4. Uniformity of Cell Shape: 1 - 10
# 5. Marginal Adhesion: 1 - 10
# 6. Single Epithelial Cell Size: 1 - 10
# 7. Bare Nuclei: 1 - 10
# 8. Bland Chromatin: 1 - 10
# 9. Normal Nucleoli: 1 - 10
# 10. Mitoses: 1 - 10
# 11. Class: (2 for benign, 4 for malignant)

dataset = pd.read_csv("bcoriginal.csv")
print(dataset)
dataset = dataset.drop(["id"], axis = 1)
print(dataset)
M = dataset[dataset.c10 == 4]
B = dataset[dataset.c10 == 2]
plt.title("Malignant vs Benign Tumor")
plt.xlabel("c1")
plt.ylabel("c2")
plt.scatter(M.c1, M.c2, color = "green", label = "Malignant", alpha = 0.8)
plt.scatter(B.c1, B.c2, color = "blue", label = "Benign", alpha = 0.8)
plt.legend()
plt.show()
dataset.c10 = [1 if i == 4 else 0 for i in dataset.c10.values]
x = dataset.drop(["c10"], axis = 1)
print(x)
y=dataset["c10"].values
print(y)
# x = (x - np.min(x)) / (np.max(x) - np.min(x))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
nb = GaussianNB()
nb.fit(x_train, y_train)
print("Naive Bayes score: ",nb.score(x_test, y_test))

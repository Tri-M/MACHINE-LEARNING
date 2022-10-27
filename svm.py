from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


dataset = datasets.load_breast_cancer()


X = dataset.data
Y = dataset.target


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


svc = svm.SVC(kernel='linear').fit(X_train, Y_train)

print(svc)
Y_predict = svc.predict(X_test)

# for i in range(len(Y_predict)):
#     print(Y_predict[i], Y_test[i])   
    

#confusion matrix
cm = confusion_matrix(Y_test, Y_predict)
print(cm)


#accuracy measures
print("Accuracy: ", accuracy_score(Y_test, Y_predict))
print("Precision: ", precision_score(Y_test, Y_predict))
print("Recall: ", recall_score(Y_test, Y_predict))

import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
from urllib.request import urlopen 

df=pd.read_csv("bc.csv")

names=['id', 'diagnosis', 'radius_mean', 
         'texture_mean', 'perimeter_mean', 'area_mean', 
         'smoothness_mean', 'compactness_mean', 
         'concavity_mean','concave_points_mean', 
         'symmetry_mean', 'fractal_dimension_mean',
         'radius_se', 'texture_se', 'perimeter_se', 
         'area_se', 'smoothness_se', 'compactness_se', 
         'concavity_se', 'concave_points_se', 
         'symmetry_se', 'fractal_dimension_se', 
         'radius_worst', 'texture_worst', 
         'perimeter_worst', 'area_worst', 
         'smoothness_worst', 'compactness_worst', 
         'concavity_worst', 'concave_points_worst', 
         'symmetry_worst', 'fractal_dimension_worst'] 

bm=['Benign', 'Malignant']
print(df.head())

df.set_index('id', inplace=True)

df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})

mapping=df['diagnosis']
print(mapping)

df.apply(lambda x:x.isnull().sum())


df.apply(lambda x:x.isnull().sum())

print(df.shape)
print(df.dtypes)

info=pd.value_counts(df.diagnosis)

benign=info[0]
malignant=info[1]

totalCases=len(df)

bPercent=benign/totalCases;
mPercent=malignant/totalCases;
print("Benign: ", bPercent, "Malignant: ", mPercent)

X=df.iloc[:,df.columns!='diagnosis']
Y=df.iloc[:,df.columns=='diagnosis']


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

Y_train=Y_train.values.ravel()
Y_test=Y_test.values.ravel()

rf=RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42)

np.random.seed(42)
start=time.time()

param_dist = {'max_depth': [2, 3, 4],
              'bootstrap': [True, False],
              'max_features': ['auto', 'sqrt', 'log2', None],
              'criterion': ['gini', 'entropy']}
cv_rf = GridSearchCV(rf, cv = 5,
                     param_grid=param_dist, 
                     n_jobs = 3)
cv_rf.fit(X_train, Y_train)
print('Best Parameters using grid search: \n', cv_rf.best_params_)                 

end=time.time()
rf.set_params(criterion = 'gini',
                  max_features = 'log2', 
                  max_depth = 3, 
                  )
        

rf.set_params(warm_start=True, 
                  oob_score=True)

min_estimators = 15
max_estimators = 500

error_rate = {}

for i in range(min_estimators, max_estimators + 1):
    rf.set_params(n_estimators=i)
    rf.fit(X_train, Y_train)

    oob_error = 1 - rf.oob_score_
    error_rate[i] = oob_error
oob_series = pd.Series(error_rate)

fig, ax = plt.subplots(figsize=(10, 10))

ax.set_facecolor('#fafafa')

oob_series.plot(kind='line',
                color = 'blue')
plt.axhline(0.055, 
            color='#875FDB',
           linestyle='--')
plt.axhline(0.05, 
            color='#875FDB',
           linestyle='--')
plt.xlabel('n_estimators')
plt.ylabel('OOB Error Rate')
plt.title('OOB Error Rate Across various Forest sizes \n(From 15 to 1000 trees)')
rf.set_params(n_estimators=420,
                  bootstrap = True,
                  warm_start=False, 
                  oob_score=False)
rf.fit(X_train, Y_train)

def variable_importance(fit):
    try:
        if not hasattr(fit, 'fit'):
            return print("'{0}' is not an instantiated model from scikit-learn".format(fit)) 

        # Checks whether model has been trained
        if not vars(fit)["estimators_"]:
            return print("Model does not appear to be trained.")
    except KeyError:
        print("Model entered does not contain 'estimators_' attribute.")

    importances = fit.feature_importances_
    
    indices = np.argsort(importances)[::-1]
    return {'importance': importances,
            'index': indices}

rf_var_imp = variable_importance(rf)

rf_importances = rf_var_imp['importance']

rf_indices = rf_var_imp['index']

def print_var_importance(importance, indices, name_index):
    print("Feature ranking:")
    
    for f in range(0, indices.shape[0]):
        i = f
        # prints the name of the feature and its importance metric 
        print("{0}. The feature '{1}' has a Mean Decrease in Impurity of {2:.5f}"
              .format(f + 1, names_index[indices[i]], importance[indices[f]]))

names_index = names[2:]


print_var_importance(rf_importances, rf_indices, names_index)

def variable_importance_plot(importance, indices, name_index):
    index = np.arange(len(names_index))

    importance_desc = sorted(importance)

    feature_space = []

    for i in range(indices.shape[0] - 1, -1, -1):
        feature_space.append(names_index[indices[i]])

    fig, ax = plt.subplots(figsize=(10, 10))

    plt.title('Feature importances for Random Forest Model\\nBreast Cancer (Diagnostic)')
    
    plt.barh(index,
              importance_desc,
              align="center",
              color = '#FFB6C1')
    plt.yticks(index,
                feature_space)

    plt.ylim(-1, 30)
    plt.xlim(0, max(importance_desc) + 0.01)
    plt.xlabel('Mean Decrease in Impurity')
    plt.ylabel('Feature')

    plt.show()
    plt.close()
variable_importance_plot(rf_importances, rf_indices, names_index)
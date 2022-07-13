import pandas as pd

data = pd.read_csv('iris.csv')

cov = data.cov()
corr = data.corr(method = 'pearson')
print('Covariance:',cov)
print("Correlation:",corr)
for i in data.columns:
    
    if i != 'class':
        print(i)
        mean = data[i].mean()
        median = data[i].median()
        std = data[i].std()
        mode = data[i].mode()
        skew = data[i].skew()
        kurt = data[i].kurt()
        

        print("The mean is ",mean)
        print("\nThe median is ",median)
        print("\nThe standard deviation is ",std)
        print("\nThe mode is ",mode)
        print("\nThe skewness is ",skew)
        print("\nThe kurtosis is ",kurt)
        
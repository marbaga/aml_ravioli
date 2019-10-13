#this script performs feature selection based on covariance
#results with current tuning scores around 0.49
#THIS FEATURE SELECTION IS OUTCLASSED BY LIBRARY METHODS

import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
import warnings
from sklearn import linear_model
from sklearn.model_selection import cross_validate


#read inliners (not stanndardized yet)
X = pd.read_csv('task1/results/X_inliners.csv', ',')

#fill nan values
filler = SimpleImputer(missing_values=np.nan, strategy='median')
X = filler.fit_transform(X)[:, 1:]
y = pd.read_csv('task1/results/y_inliners.csv', ',').values[:, 1]
X = pd.DataFrame(X)
print(X)

#insert y column
X.insert(X.shape[1], 'y', y, True)

#eliminate features with variance 0
sel = VarianceThreshold(threshold=0)
X = sel.fit_transform(X)

#feature selection
cor = pd.DataFrame(X).corr()
columns = np.full((cor.shape[0],), True, dtype=bool)
for i in range(cor.shape[0]):
    for j in range(i+1, cor.shape[0]):
        if abs(cor.iloc[i,j]) > 0.95 or abs(cor.iloc[j,cor.shape[0]-1]) < 0.05:
            if columns[j]:
                columns[j] = False
selected_columns = pd.DataFrame(X).columns[columns]
data = pd.DataFrame(X[:, selected_columns])

#removing y
data = data.iloc[:, :data.shape[1]-1]

print(data)

poly = preprocessing.PolynomialFeatures(2)
#data = poly.fit_transform(data)  #uncomment to try out a 2nd order polynomial model
data = pd.DataFrame(data)
print (data)

#uncomment to do further feature selection on transformed features
'''
data.insert(data.shape[1], 'y', y, True)
print(data.shape)

cor2 = pd.DataFrame(data).corr()

columns2 = np.full((cor2.shape[0],), True, dtype=bool)
print('Computed corr matrix')
for i in range(cor2.shape[0]):
    for j in range(i+1, cor2.shape[0]):
        if abs(cor2.iloc[i,j]) >= 0.9 or abs(cor2.iloc[j,cor2.shape[0]-1]) < 0.001:
            if columns2[j]:
                columns2[j] = False
selected_columns2 = pd.DataFrame(data).columns[columns2]
print(selected_columns2.shape)
data2 = pd.DataFrame(data).loc[:, selected_columns2]

print(data2.shape)

print('Done')
'''

#suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#look at score using all features
print("Old score")
X_t = pd.read_csv('task1/results/X_inliners.csv', ',').iloc[:, 1:]
scaler = preprocessing.StandardScaler()
X_t = pd.DataFrame(scaler.fit_transform(X_t))
print(X_t)

y_t = y

#play around with a linear model to find best hyperparameters
model = linear_model.ElasticNet()
cv_results = cross_validate(model, X_t, y_t, cv=10)
print(cv_results['test_score'])
print("Average: " + str(np.average(cv_results['test_score'])))

#standardize
scaler = preprocessing.StandardScaler()
X_t = scaler.fit_transform(data)
print(X_t)

#ElasticNetCV performs automatic hyperparameter search. Lasso models works a little better.
#we should try different linear models
model = linear_model.RidgeCV(cv=5)
model.fit(X_t, y_t.ravel())
print(model.score(X_t, y_t))
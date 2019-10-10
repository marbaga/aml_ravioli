#RavioliRegress AML 2019

#All the code has to be rewritten properly and optimized

import warnings
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model, preprocessing
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PolynomialFeatures

#suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#load preprocessed data
X_t = pd.read_csv('task1/results/X_inliners.csv', ',').iloc[:, 2:].values
y_t = pd.read_csv('task1/results/y_inliners.csv', ',').iloc[:, 2:].values

print(y_t)
# TODO: feature selection

# First of all we observe the effects of standardization/regularization on linear models
# Most likely the standardized model has a slightly better performance

scaler = preprocessing.StandardScaler().fit(X_t)
X_t_s = X_t
X_t_s = scaler.transform(X_t_s) # X_t_s has now been scaled

#print(X_t)
#print(X_t_s)
'''
regr = linear_model.LinearRegression()
cv_results = cross_validate(regr, X_t, y_t, cv=10)

print("Evaluating effects of standardization")
print("Score without standardization")
print(cv_results['test_score'])
print("Average: " + str(np.average(cv_results['test_score'])))
print()

cv_results = cross_validate(regr, X_t_s, y_t, cv=10)
print("Score with standardization")
print(cv_results['test_score'])
print("Average: " + str(np.average(cv_results['test_score'])))
print()
'''
# Then we train each model (different parameters) and compute the cross validation score (with or without standardization
print("Trying out linear models")

arr = [linear_model.LinearRegression()]#[linear_model.LinearRegression(), linear_model.Lasso(), linear_model.Ridge(), linear_model.ElasticNet()]
'''
for model in arr:
    cv_results = cross_validate(model, X_t_s, y_t, cv=5)
    print('Score of ' + str(model) + ': ')
    print(cv_results['test_score'])
    print("Average: " + str(np.average(cv_results['test_score'])))
    print()
'''
for i in range (1, 5):
    for model in arr:
        print("Trying out polynomial models of degree " + str(i))
        poly = PolynomialFeatures(i)
        X_t_m = X_t_s[:, 1:150]
        poly.fit(X_t_m)
        X_t_m = poly.transform(X_t_m)
        print(str(len(X_t_m)) + ' ' + str(len(X_t_m[0])))
        cv_results = cross_validate(model, X_t_m, y_t, cv=5)
        print('Score of ' + str(model) + 'of degree ' + str(i) + ': ')
        print(cv_results['test_score'])
        print("Average: " + str(np.average(cv_results['test_score'])))
        print()

# We can now choose the right model and adjust the hyperparameters
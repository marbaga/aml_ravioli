#RavioliRegress AML 2019

#All the code has to be rewritten properly and optimized
#THIS SCRIPT HAS BEEN USED TO COMPARE VARIOUS MODELS. IT IS NOW OUTDATED

import warnings
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PolynomialFeatures

#suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#load preprocessed data
X_t = pd.read_csv('task1/results/X_normalized.csv', ',').iloc[:, 1:].values #removing indices and casting
y_t = pd.read_csv('task1/results/y_inliners.csv', ',').iloc[:, 1:].values #removing indices and casting

# TODO: feature selection

# We train each model (different parameters) and compute the cross validation score (with or without standardization
print("Trying out linear models")

arr = [linear_model.LinearRegression(), linear_model.Lasso(), linear_model.Ridge(), linear_model.ElasticNet()]
for i in range (1, 5):
    for model in arr:
        print("Trying out linear models of degree " + str(i))
        poly = PolynomialFeatures(i)
        X_t_m = X_t
        poly.fit(X_t_m)
        X_t_m = poly.transform(X_t_m)   #creating olynomial features
        cv_results = cross_validate(model, X_t_m, y_t, cv=10)
        print('Score of ' + str(model) + 'of degree ' + str(i) + ': ')
        print(cv_results['test_score'])
        print("Average: " + str(np.average(cv_results['test_score'])))
        print()

# Training 2nd degree polynomial models already takes too much time, features should be reduced to around 150
# The best results are obtained by algorithms using L1 norm, feature reduction can help a lot
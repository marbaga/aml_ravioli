#RavioliRegress AML 2019

#All the code has to be rewritten properly and optimized

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

#play around with a linear model to find best hyperparameters
model = linear_model.ElasticNet(alpha=0.45, l1_ratio=0.8)
cv_results = cross_validate(model, X_t, y_t, cv=10)
print(cv_results['test_score'])
print("Average: " + str(np.average(cv_results['test_score'])))

#Automatic CV or parameter selection
#model = linear_model.ElasticNetCV(cv=5, max_iter=2000)
#model = linear_model.LassoCV(cv=5)
#model.fit(X_t, y_t.ravel())
#print(model.score(X_t, y_t))


print()

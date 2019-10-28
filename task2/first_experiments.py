import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from scipy.stats import shapiro, boxcox
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import ensemble
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import svm


class CustomEstimator (BaseEstimator):

    def __init__(self,
                 model=GradientBoostingClassifier(),
                 balance_by_deletion=True,
                 ):
        self.model = model
        self.balance_by_deletion = balance_by_deletion
        self.indices = []


    def fit(self, X, y):

        X_t = X.copy()
        y_t = y.copy()

        if self.balance_by_deletion:
            min = len(y_t)

            unique, counts = np.unique(y_t, return_counts=True)
            min = np.min(counts)
            self.indices = []
            for i in unique:
                draw = np.random.choice(np.where(y_t == i)[0], min)
                for j in draw:
                    self.indices.append(j)
            X_t = X_t[self.indices, :]
            y_t = y_t[self.indices]

        print('Final training matrix shape is ' + str(X_t.shape))

        self.model.fit(X_t, y_t)
        print('Finished fitting')

        return self

    def predict(self, X):

        X_t = X.copy()

        predictions = self.model.predict(X_t)

        return predictions

#This bit performs cross validation. Every transformation on data needs to be carried out inside the custom estimator class. The following lines should not be touched
X_t = pd.read_csv('X_train.csv', ',').iloc[:, 1:].to_numpy()
y_t = pd.read_csv('y_train.csv', ',').iloc[:, 1].to_numpy()
X_test = pd.read_csv('X_test.csv', ',').iloc[:, 1:].to_numpy()


model = CustomEstimator()
cv_results = cross_validate(model, X_t, y_t, scoring='balanced_accuracy', n_jobs=-1, cv=10, verbose=True)
print('Score of ' + str(model) + ': ')
print(cv_results['test_score'])
print("Average: " + str(np.average(cv_results['test_score'])))
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from functools import partial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from scipy.stats import shapiro, boxcox
from sklearn.feature_selection import VarianceThreshold, RFE, RFECV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import ensemble
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import imblearn
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dropout, BatchNormalization, Activation
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Conv1D, MaxPool1D
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from keras import optimizers, regularizers
from sklearn.model_selection import GridSearchCV

class CustomEstimator (BaseEstimator):

    def __init__(self, max_depth=3, learning_rate=0.09, n_estimators=69):
        self.model = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_t = X.copy()
        y_t = y.copy()
        X_t = self.scaler.fit_transform(X_t)
        self.model.fit(X_t, y_t)
        return self

    def predict(self, X):
        X_t = X.copy()
        X_t = self.scaler.transform(X_t)
        return self.model.predict(X_t)

X_t = pd.read_csv('extracted_features.csv', ',').iloc[:, 1:].to_numpy()
y_t = pd.read_csv('y_train.csv', ',').iloc[:, 1].to_numpy()

list = pd.isnull(pd.DataFrame(X_t)).any(1).to_numpy().nonzero()[0]
X_t=np.delete(X_t, list, axis=0)
y_t=np.delete(y_t, list, axis=0)

parameters = {
    'n_estimators': [500],
}

grid_search = GridSearchCV(
    estimator=XGBClassifier(max_depth=10, objective='multi:softmax', n_estimators=500),
    param_grid=parameters,
    scoring = 'f1_micro',
    n_jobs = -1,
    cv = 10,
    verbose=True
)

grid_search.fit(X_t, y_t)

print(grid_search.best_estimator_)
print(grid_search.best_params_)
print(grid_search.best_score_)

#cv_results = cross_validate(model, X_t, y_t, scoring='f1_micro', n_jobs=-1, cv=10, verbose=True)
#print('Score of ' + str(model) + ': ')
#print(cv_results['test_score'])
#print("Average: " + str(np.average(cv_results['test_score'])))
#print("Variance: " + str(np.var(cv_results['test_score'])))

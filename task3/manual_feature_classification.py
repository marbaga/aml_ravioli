#try class weights

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

class CustomEstimator (BaseEstimator):

    def __init__(self, sampler, model, scale, sample):
        self.model = model
        self.indices = []
        self.sampler = sampler
        self.scaler = StandardScaler()
        self.scale=scale
        self.sample=sample

    def fit(self, X, y):
        X_t = X.copy()
        y_t = y.copy()
        if self.sample:
            X_t, y_t = self.sampler.fit_sample(X_t, y_t)
        if self.scale:
            X_t = self.scaler.fit_transform(X_t)
        self.model.fit(X_t, y_t)
        return self

    def predict(self, X):
        X_t = X.copy()
        if self.scale:
            X_t = self.scaler.transform(X_t)
        return self.model.predict(X_t)

def baseline_model():
    model = Sequential()
    model.add(Dense(100, input_dim=56))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='softmax'))
    opt = optimizers.Adamax(learning_rate=0.003)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    return model

#This bit performs cross validation. Every transformation on data needs to be carried out inside the custom estimator class. The following lines should not be touched
X_t = pd.read_csv('X_train_manual_extraction.csv', ',').iloc[:, 1:].to_numpy()
y_t = pd.read_csv('y_train.csv', ',').iloc[:, 1].to_numpy()
X_test_manual_extraction = pd.read_csv('X_test_manual_extraction.csv', ',').iloc[:, 1:].to_numpy()

#list = pd.isnull(pd.DataFrame(X_t)).any(1).to_numpy().nonzero()[0]
#X_t=np.delete(X_t, list, axis=0)
#y_t=np.delete(y_t, list, axis=0)

inner_models = [RandomForestClassifier(),
                    ExtraTreesClassifier(), #752
                    KerasClassifier(build_fn=baseline_model, epochs=80, batch_size=64, verbose=2), #800
                    LogisticRegression(), #770
                    SVC(), #776
                    XGBClassifier(), #799
                    GradientBoostingClassifier(), #799
                    XGBClassifier(max_depth=10, objective='multi:softmax', n_estimators=300), #808
                    MLPClassifier()] #790

model = CustomEstimator(sampler=imblearn.under_sampling.RandomUnderSampler(), model= XGBClassifier(max_depth=10, objective='multi:softmax', n_estimators=300), sample=False, scale=True)

#cv_results = cross_validate(model, X_t, y_t, scoring='f1_micro', n_jobs=-1, cv=10, verbose=True)
#print('Score of ' + str(model) + ': ')
#print(cv_results['test_score'])
#print("Average: " + str(np.average(cv_results['test_score'])))
#print("Variance: " + str(np.var(cv_results['test_score'])))

model.fit(X_t, y_t)
test_pred = model.predict(X_test_manual_extraction)
answer = pd.read_csv('X_test.csv', ',')[['id']]
answer = pd.concat([answer, pd.DataFrame(data=test_pred, columns=['y'])], axis=1)
pd.DataFrame(answer).to_csv('result_xgb.csv', ',', index=False)
'''
Neural network fitting the entire dataset
'''

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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
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
from keras.layers import Dropout
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Conv1D, MaxPool1D
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from keras import optimizers, regularizers
class CustomEstimator (BaseEstimator):

    def __init__(self, sampler,
                 model
                 ):
        self.model = model
        self.indices = []
        self.sampler = sampler
        self.scaler = StandardScaler()

    def fit(self, X, y):

        X_t = X.copy()
        y_t = y.copy()
        #X_t, y_t = self.sampler.fit_sample(X_t, y_t)

        #self.feature_selector = PCA(n_components=500)
        #self.feature_selector.fit(X_t)
        #X_t = self.feature_selector.transform(X_t)

        '''
        for i in range(0, X_t.shape[1]):
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
            sns.boxplot(X_t[np.where(y_t == 0), i], color='blue', ax=ax1)
            sns.boxplot(X_t[np.where(y_t == 1), i], color='red', ax=ax2)
            sns.boxplot(X_t[np.where(y_t == 2), i], color='pink', ax=ax3)
            plt.show()
            '''
        print('Final training matrix shape is ' + str(X_t.shape))
        X_t = self.scaler.fit_transform(X_t)
        self.model.fit(X_t, y_t, class_weight={0:1, 1:0.16, 2:1})

        return self

    def predict(self, X):

        X_t = X.copy()
        #X_t = self.feature_selector.transform(X_t)
        X_t = self.scaler.transform(X_t)
        predictions = self.model.predict(X_t)

        return predictions

def baseline_model():
    model = Sequential()
    model.add(Dense(700, input_dim=1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(400, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    opt = optimizers.Adam(lr=0.0002)

    model.compile(loss='categorical_crossentropy', optimizer=opt)
    print(model.summary())
    return model


#This bit performs cross validation. Every transformation on data needs to be carried out inside the custom estimator class. The following lines should not be touched
X_t = pd.read_csv('X_train.csv', ',').iloc[:, 1:].to_numpy()
y_t = pd.read_csv('y_train.csv', ',').iloc[:, 1].to_numpy()
X_test = pd.read_csv('X_test.csv', ',').iloc[:, 1:].to_numpy()
model = CustomEstimator(sampler=imblearn.under_sampling.RandomUnderSampler(), model= KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=32, verbose=1))#
cv_results = cross_validate(model, X_t, y_t, scoring='balanced_accuracy', n_jobs=-1, cv=10, verbose=True)
print('end')
print('Score of ' + str(model) + ': ')
print(cv_results['test_score'])
print("Average: " + str(np.average(cv_results['test_score'])))
print("Variance: " + str(np.var(cv_results['test_score'])))

model.fit(X_t, y_t)
pred = model.predict(X_test)
answer = pd.read_csv('X_test.csv', ',')[['id']]
answer = pd.concat([answer, pd.DataFrame(data=pred, columns=['y'])], axis=1)
pd.DataFrame(answer).to_csv('result.csv', ',', index=False)
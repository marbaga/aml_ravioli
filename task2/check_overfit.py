'''
Experimenting with visualization of loss through epochs
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras import regularizers

from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
import keras.backend as K

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
    model.add(Dense(50, input_dim=1000))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(400, kernel_regularizer=regularizers.l2(0.0)))
    model.add(Dense(3, activation='softmax'))
    opt = optimizers.Adam(lr=0.0002)

    model.compile(loss='categorical_crossentropy', optimizer=opt)
    print(model.summary())
    return model

def plot_metrics(history):
    metrics = ['loss']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.ylim([0, plt.ylim()[1]])
        plt.legend()
    plt.show()

#This bit performs cross validation. Every transformation on data needs to be carried out inside the custom estimator class. The following lines should not be touched
X_t = pd.read_csv('X_train.csv', ',').iloc[:, 1:].to_numpy()
y_t = pd.read_csv('y_train.csv', ',').iloc[:, 1].to_numpy()
#X_test = pd.read_csv('X_test.csv', ',').iloc[:, 1:].to_numpy()

y_t = np.transpose(np.atleast_2d(y_t))
X_t = np.concatenate([X_t, y_t], axis=1)

np.random.shuffle(X_t)

X_train, X_test = train_test_split(X_t, test_size=0.1)
X_train, X_val = train_test_split(X_train, test_size=0.1)

y_train_copy = X_train[:, -1].ravel()
y_train = np_utils.to_categorical(X_train[:, -1].ravel())
y_val = np_utils.to_categorical(X_val[:, -1].ravel())
y_test = X_test[:, -1].ravel()
X_train = X_train[:, :X_train.shape[1]-1]
X_val = X_val[:, :X_val.shape[1]-1]
X_test = X_test[:, :X_test.shape[1]-1]
X_train_copy = X_train

#sampler = imblearn.over_sampling.SMOTE()
sampler = imblearn.over_sampling.SMOTE()
X_train, y_train = sampler.fit_sample(X_train, y_train)

scaler = StandardScaler()
scaler.fit(X_train)
scaler.transform(X_train)
scaler.transform(X_val)
scaler.transform(X_test)

model = baseline_model()
#model.load_weights(initial_weights)
baseline_history = model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=10,
    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.001, patience=32)],
    validation_data=(X_val, y_val))

pred = np.argmax(model.predict(X_test), axis=1)
print(balanced_accuracy_score(y_test, pred))
print(balanced_accuracy_score(y_train_copy, np.argmax(model.predict(X_train_copy), axis=1)))
plot_metrics(baseline_history)
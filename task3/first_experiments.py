import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import cross_validate
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from sklearn.base import BaseEstimator
from keras.wrappers.scikit_learn import KerasClassifier
import imblearn
from keras.layers import Conv1D, Dropout, BatchNormalization, MaxPool1D
import sklearn.utils
from scipy import stats



class CustomEstimator (BaseEstimator):

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        X_t = X.copy()
        X_t = X_t.reshape(X_t.shape[0], X_t.shape[1], 1)
        print(X_t.shape)
        y_t = y.copy()
        class_weight = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(y_t), y_t)
        self.model.fit(X_t, y_t, class_weight=class_weight)
        return self

    def predict(self, X):
        X_t = X.copy()
        X_t.reshape(X_t.shape[0], X_t.shape[1], 1)
        predictions = self.model.predict(X_t)
        return predictions


def baseline_model():
    model = Sequential()
    model.add(Conv1D(16, kernel_size=50, activation='relu', input_shape=(2500,1)))
    model.add(Conv1D(16, kernel_size=50, activation='relu'))
    model.add(MaxPool1D(pool_size=3))
    model.add(Dropout(0.1))
    model.add(Conv1D(32, kernel_size=10, activation='relu'))
    model.add(Conv1D(32, kernel_size=10, activation='relu'))
    model.add(MaxPool1D(pool_size=3))
    model.add(Dropout(0.1))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPool1D(pool_size=3))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    return model

print('Starting')
X_t = pd.read_csv('X_train.csv', ',').iloc[:, 1:].to_numpy()
y_t = pd.read_csv('y_train.csv', ',').iloc[:, 1].to_numpy()
#X_test = pd.read_csv('X_test.csv', ',').iloc[:, 1:].to_numpy()
print('Read')
#print(X_t)
#print(y_t)

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

for i in range(0,X_t.shape[1]):
    print(signaltonoise(X_t[:, i]))
    print(y_t[i])

model = CustomEstimator(model=KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=128, verbose=1))
model.fit(X_t, y_t)
cv_results = cross_validate(model, X_t, y_t, scoring='f1_micro', n_jobs=-1, cv=10, verbose=True)

print('Score of ' + str(model) + ': ')
print(cv_results['test_score'])
print("Average: " + str(np.average(cv_results['test_score'])))
print("Variance: " + str(np.var(cv_results['test_score'])))


'''
#pd.set_option('display.max_rows', len(X_t))
#print(X_t.isnull().sum(axis=1))
for i in [0,1,2,3,4,5,6,7,8,9,10]:
    X_t.iloc[i].plot()
#plt.hist(y_t)
#print(X_test)
#print(X_t)
#plt.hist(X_test.isnull().sum(axis=1), bins=1000)
    plt.show()
#plt.hist(X_t.isnull().sum(axis=1), bins=1000)
#plt.show()
print('Done')
'''
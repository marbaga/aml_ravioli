'''
Experimenting with visualization of loss through epochs
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
import imblearn
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Conv1D, MaxPool1D, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.initializers import Constant
from keras import regularizers
import time

from keras import backend as K
from keras.layers.normalization import BatchNormalization

def baseline_model():
    model = Sequential()
    model.add(Dense(100, input_dim=56))
    #model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    #model.add(Dense(400, kernel_regularizer=regularizers.l2(0.0)))
    model.add(Dense(4, activation='softmax'))
    #opt = optimizers.Adam(lr=0.0002), adagrad, adamax
    #opt = optimizers.Adamax(learning_rate=0.003) #0.002 is also fine, 100 neurons, 0.2 dropout, 80 epochs, batch size 64
    #opt = optimizers.Adagrad(learning_rate=0.01) #0.2 dropout, 100 neurons, 100 epochs, batch size 64 - SLIGHTLY WORSE THAN ADAMAX
    #opt = optimizers.Adam(learning_rate=0.001) #0.2 dropout, 100 neurons, 100 epochs, batch size 64 - pretty bad
    opt = optimizers.Adamax(learning_rate=0.003)
    model.load_weights("weights.best.hdf5")
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics = ['accuracy'])
    #print(model.summary())
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

past_scores = []
#for i in range(0,10):
#This bit performs cross validation. Every transformation on data needs to be carried out inside the custom estimator class. The following lines should not be touched
X_t = pd.read_csv('X_train_manual_extraction.csv', ',').iloc[:, 1:].to_numpy()
y_t = pd.read_csv('y_train.csv', ',').iloc[:, 1].to_numpy()

X_test_manual_extraction = pd.read_csv('X_test_manual_extraction.csv', ',').iloc[:, 1:].to_numpy()

#list = pd.isnull(pd.DataFrame(X_t)).any(1).to_numpy().nonzero()[0]
#X_t=np.delete(X_t, list, axis=0)
#y_t=np.delete(y_t, list, axis=0)

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

scaler = StandardScaler()
scaler.fit(X_train)
scaler.transform(X_train)
scaler.transform(X_val)
scaler.transform(X_test)

model = baseline_model()
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
'''
baseline_history = model.fit(
        X_train,
        y_train,
        batch_size=64,
        epochs=80, #100-120 is more appropriate
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.001, patience=32)],#, checkpoint],#32
        validation_data=(X_val, y_val), verbose=1)
'''
pred = np.argmax(model.predict(X_test), axis=1)
print(f1_score(y_test, pred, average='micro'))
print(f1_score(y_train_copy, np.argmax(model.predict(X_train_copy), axis=1), average='micro'))
#plot_metrics(baseline_history)

scaler.transform(X_test_manual_extraction)
test_pred = np.argmax(model.predict(X_test_manual_extraction), axis=1)
answer = pd.read_csv('X_test.csv', ',')[['id']]
answer = pd.concat([answer, pd.DataFrame(data=test_pred, columns=['y'])], axis=1)
pd.DataFrame(answer).to_csv('result_nn.csv', ',', index=False)
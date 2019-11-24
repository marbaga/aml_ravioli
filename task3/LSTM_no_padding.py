from keras import Sequential
from keras.utils import Sequence
from keras.layers import LSTM, Dense, Masking
import numpy as np
import pandas as pd
from biosppy import ecg
from keras import Sequential, regularizers
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Input
from keras.models import Model, load_model
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.model_selection import cross_validate
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator


class MyBatchGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, X, y, batch_size=1, shuffle=False):
        'Initialization'
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.y)/self.batch_size))

    def __getitem__(self, index):
        return self.__data_generation(index)

    def on_epoch_end(self):
        'Shuffles indexes after each epoch'
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        Xb = np.empty((self.batch_size, *self.X[index].shape))
        yb = np.empty((self.batch_size, *self.y[index].shape))
        # naively use the same sample over and over again
        for s in range(0, self.batch_size):
            Xb[s] = self.X[index]
            yb[s] = self.y[index]
        return Xb, yb

class MyTestGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, X, batch_size=1, shuffle=False):
        'Initialization'
        self.X = X
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        return self.__data_generation(index)

    def on_epoch_end(self):
        'Shuffles indexes after each epoch'
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        Xb = np.empty((self.batch_size, *self.X[index].shape))
        # naively use the same sample over and over again
        for s in range(0, self.batch_size):
            Xb[s] = self.X[index]
        return Xb

'''
# Estimation class for LSTM (necessary for cross validation)
class CustomNN(BaseEstimator):

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit_generator(MyBatchGenerator(X_t, y_t, batch_size=1), epochs=5, use_multiprocessing=False)
        return self

    def predict(self, X):
        predictions = self.model.predict_generator(MyTestGenerator(X_t), verbose=1, use_multiprocessing=False)
        y_p = np.array([])
        for y in predictions:
            y_p = np.append(y_p, np.argmax(y))
        return y_p
    '''

# LSTM model
def baseline_model_sample():
    model = Sequential()
    model.add(LSTM(units=30, input_shape=(None, 30), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(units=30, input_shape=(None, 30), return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# COMPUTATION
X_t = pd.read_csv('X_train.csv', ',').iloc[:, 1:].to_numpy()
y_t = pd.read_csv('y_train.csv', ',').iloc[:, 1].to_numpy()

print('X_t shape: ' + str(X_t.shape))
print('y_t shape: ' + str(y_t.shape))

loaded_model = load_model('autoencoder.h5')

encoder = Model(loaded_model.input, loaded_model.layers[-3].input)  # Input(shape=(180,))

encoded_matrix = []

for row in X_t:
    decomposed_row = np.empty([1, 180])  # initializing meaningless line
    row = pd.Series(row)
    row = row.dropna().values
    _, _, _, _, templates, _, _ = ecg.ecg(signal=row, sampling_rate=300.0, show=False)
    decomposed_row = np.concatenate((decomposed_row, templates), axis=0)
    decomposed_row = decomposed_row[1:]  # removing meaningless line
    decomposed_encoded_row = []
    for beat in decomposed_row:
        decomposed_encoded_row.append(encoder.predict(np.array([beat]))[0])
    encoded_matrix.append(np.array(decomposed_encoded_row))

encoded_matrix = np.array(encoded_matrix)
print(encoded_matrix.shape)
print('Computed encoded matrix')


X_train, X_test, y_train, y_test = train_test_split(encoded_matrix, y_t, test_size=0.2)

# preparing y_t
y_train = to_categorical(y_train, num_classes=4)

baseline_model_sample().fit_generator(MyBatchGenerator(X_train, y_train, batch_size=1), epochs=5)     # , use_multiprocessing=True

y_pred = baseline_model_sample().predict_generator(MyTestGenerator(X_test), verbose=1)   #, use_multiprocessing=True

y_p = np.array([])
for y in y_pred:
    y_p = np.append(y_p, np.argmax(y))

F1 = f1_score(y_test, y_p, average='micro')
print('Score: ' + str(F1))

'''

# cross validation
model = CustomNN(baseline_model_sample())
# evaluate model
cv_results = cross_validate(model, X_t, y_t, scoring='f1_micro', cv=5, verbose=True)
print('Score of ' + str(model) + ': ')
print(cv_results['test_score'])
print("Average: " + str(np.average(cv_results['test_score'])))
print("Variance: " + str(np.var(cv_results['test_score'])))'''
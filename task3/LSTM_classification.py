import pandas as pd
import numpy as np
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


X_t = pd.read_csv('X_train.csv', ',').iloc[:, 1:].to_numpy()
y_t = pd.read_csv('y_train.csv', ',').iloc[:, 1].to_numpy()

print(X_t.shape)
print(y_t.shape)

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
        decomposed_encoded_row.append(encoder.predict(np.array([beat]))[0].tolist())
    encoded_matrix.append(decomposed_encoded_row)

empty_list = np.zeros(30).tolist()
# finding maximum length of sequence
max_sequence = 0
for row in encoded_matrix:
    if len(row) > max_sequence:
        max_sequence = len(row)

for row in encoded_matrix:
    while (len(row) < max_sequence):
        row.append(empty_list)

encoded_matrix = np.array(encoded_matrix)
print(encoded_matrix.shape)
print('Computed encoded matrix')

X_train, X_test, y_train, y_test = train_test_split(encoded_matrix, y_t, test_size=0.2)

# preparing y_t
y_train = to_categorical(y_train, num_classes=4)

# fit and evaluate a model
model = Sequential()
model.add(LSTM(units=30, input_shape=(None, 30), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(units=30, input_shape=(None, 30), return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit network
history = model.fit(X_train, y_train, epochs=15, batch_size=64, shuffle=True,
                validation_split=0.2, verbose=True)

y_pred = model.predict(X_test, batch_size=64, verbose=1)
y_p = np.array([])
for y in y_pred:
    y_p = np.append(y_p, np.argmax(y))

F1 = f1_score(y_test, y_p, average='micro')
print('Score: ' + str(F1))

'''
# plotting accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()'''


'''
# cross validation
model = KerasClassifier(build_fn=model, epochs=64, batch_size=256, verbose=1)
# evaluate model
cv_results = cross_validate(model, X_t, y_t, scoring='f1_micro',
                            fit_params={'epochs': 20, 'batch_size': 64,
                                        'verbose': True, 'shuffle': True,},
                            cv=5, verbose=True)
print(cv_results)'''

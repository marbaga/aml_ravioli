import pandas as pd
import numpy as np
from biosppy import ecg
from keras.layers import Dense, Input, Dropout
from keras.models import Model
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

#Experimenting with autoencoders to perform automatic feature extraction

X_t = pd.read_csv('X_train.csv', ',').iloc[1:].to_numpy()

X_t = X_t.copy()
# print('Original X_train shape:')
# print(X_t.shape)

beats_matrix = np.empty([1,180])    # initializing meaningless line

for row in X_t:
    row = pd.Series(row)
    row = row.dropna().values
    _, _, _, _, templates, _, _ = ecg.ecg(signal=row, sampling_rate=300.0, show=False)
    beats_matrix = np.concatenate((beats_matrix, templates), axis=0)

beats_matrix = beats_matrix[1:]   # removing meaningless line

sequence_length = beats_matrix[0].size  # it's 180

#adding random noise to input
noise_factor = 4
beats_matrix_noisy = beats_matrix + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=beats_matrix.shape)
x_train_noisy = np.clip(beats_matrix_noisy, 0., 1.)

#AUTOENCODER

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
cb_list = [es]

# this is our input placeholder
input_sequence = Input(shape=(sequence_length,))
# "encoded" is the encoded representation of the input
encoded = Dense(90, activation='relu')(input_sequence)  # activity_regularizer=regularizers.l1(0.00001)
encoded = Dense(60, activation='relu')(encoded)
encoded = Dense(30, activation='relu')(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(60, activation='relu')(encoded)
decoded = Dense(90, activation='relu')(decoded)
decoded = Dense(180, activation='linear')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_sequence, decoded)

encoder = Model(input_sequence, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(30,))
# retrieve the last layer of the autoencoder model
deco = autoencoder.layers[-3](encoded_input)
deco = autoencoder.layers[-2](deco)
deco = autoencoder.layers[-1](deco)

# create the decoder model
decoder = Model(encoded_input, deco)

autoencoder.compile(optimizer='adam', metrics=['acc'], loss='mse')

history = autoencoder.fit(beats_matrix_noisy, beats_matrix,
                epochs=85,
                batch_size=256,
                shuffle=True,
                validation_split=0.2,
                verbose=True,
                callbacks=cb_list)

encoded_series = encoder.predict(beats_matrix)
decoded_series = decoder.predict(encoded_series)

# plotting accuracy and loss
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
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
plt.show()

# save model and architecture to file
autoencoder.save("autoencoder.h5")
print("Saved model to disk")

'''
#plotting original and decoded signal
for i in range(1000):
    decod_series = pd.Series(decoded_series[i])
    original = pd.Series(beats_matrix[i])
    #noisy_original = pd.Series(beats_matrix_noisy[i])
    plt.subplot(1, 2, 1)
    original.plot()
    plt.title('Original')
    plt.subplot(1, 2, 2)
    decod_series.plot()
    #noisy_original.plot()
    plt.title('Decoded')
    plt.show()'''

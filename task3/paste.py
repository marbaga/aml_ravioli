import pandas as pd
from scipy import stats
import numpy as np

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, Flatten, GlobalAveragePooling1D, \
    concatenate
from keras.utils import to_categorical
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
X, X_test = train_test_split(pd.read_csv('X_train.csv', ',').iloc[:, 1:2001].to_numpy(), test_size=0.1)
Y, Y_test = train_test_split(pd.read_csv('y_train.csv', ',').iloc[:, 1].to_numpy(), test_size=0.1)
X = np.expand_dims(X, axis=2)
X_test = np.expand_dims(X_test, axis=2)
print(np.isnan(X).any())
Y2=Y.copy()
Y=to_categorical(Y)
print(Y.shape)
Y2_test = Y_test.copy()
Y_test=to_categorical((Y_test))
def get_model():
    nclass = 4
    inp = Input(shape=(X.shape[1], 1))
    img_1 = Convolution1D(16, kernel_size=50, activation=activations.relu, padding="valid")(inp)
    img_1 = Convolution1D(16, kernel_size=50, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
#    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=30, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=30, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
#    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=30, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=30, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
#    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(256, kernel_size=30, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(256, kernel_size=30, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
#    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass, activation='softmax', name="dense_3_mitbih")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    return model

model = get_model()
file_path = "baseline.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early

model.fit(X, Y, epochs=1000, verbose=True, callbacks=callbacks_list, validation_split=0.1)
model.load_weights(file_path)

pred_test = model.predict(X_test)
pred_test = np.argmax(pred_test, axis=-1)

f1 = f1_score(Y2_test, pred_test, average="micro")

print("Test f1 score : %s "% f1)

acc = accuracy_score(Y2_test, pred_test)

print("Test accuracy score : %s "% acc)
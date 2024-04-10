import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os

checkpoint_path = './models/checkpoints/MHG.ckpt'
cp_dir = os.path.dirname(checkpoint_path)
def modelDefinition():

    model = Sequential()

    # adding a Dense input layer
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    # adding an output layer
    model.add(Dense(16, kernel_initializer='normal'))
    #compiling the model using the optimizer Adam and the loss function MSE
    model.compile(loss="mean_squared_error", optimizer='adam')

    return model

def trainModel(x_train, y_train,x_test, y_test):

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose= 1)

    model = modelDefinition()
    model.fit(x_train, y_train, epochs=10, batch_size=16, validation_data=(x_test,y_test), verbose=2, callbacks=[cp_callback])

    return model





import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os

checkpoint_path = 'checkpoints/MHG.ckpt'
cp_dir = os.path.dirname(checkpoint_path)
def modelDefinition():

    model = Sequential(
        # adding a Dense input layer
        Dense(8, input_dim=13, kernel_initializer='normal', activation='relu'),
        # adding an output layer
        Dense(8, kernel_initializer='normal'),
    )

    #compiling the model using the optimizer Adam and the loss function MSE
    model.compile(loss='mse', optimizer='adam')

    return model

def trainModel(x_train, y_train):

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose= 1)

    model = modelDefinition()
    model.fit(x_train, y_train, epochs= 100, validation_data=(x_train, y_train), callbacks=[cp_callback])
    return model





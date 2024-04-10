import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

def modelDefinition():

    model = Sequential()

    #adding a Dense input layer
    model.add(Dense(8, input_dim=13, kernel_initializer='normal', activation='relu'))

    #adding an output layer
    model.add(Dense(8, kernel_initializer='normal'))

    #compiling the model using the optimizer Adam and the loss function MSE
    model.compile(loss='mse', optimizer='adam')

    return model






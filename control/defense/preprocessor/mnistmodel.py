import scipy.io as sio
import numpy as np
import keras
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Dropout
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import tensorflow as tf

# Construct model
def mnist_model(logits=False, input_ph=None, img_rows=28, img_cols=28,
              channels=1, nb_classes=10):
  
    model = Sequential()

    input_shape = (img_rows, img_cols, channels)

    layers = [Conv2D(32, (5, 5), padding="same",input_shape=input_shape),
              Activation('relu'),
              MaxPooling2D(pool_size=(2,2),strides=2),
              
              Conv2D(64, (5, 5), padding="same"),
              Activation('relu'),
              MaxPooling2D(pool_size=(2,2),strides=2),

              Flatten(),
              Dense(1024),
              Activation('relu'),
              Dropout(.4),
              
              Dense(nb_classes)
              ]

    for layer in layers:
        model.add(layer)

    if logits:
        logits_tensor = model(input_ph)
    model.add(Activation('softmax'))

    if logits:
        return model, logits_tensor
    else:
        return model

#mnist_model().summary()

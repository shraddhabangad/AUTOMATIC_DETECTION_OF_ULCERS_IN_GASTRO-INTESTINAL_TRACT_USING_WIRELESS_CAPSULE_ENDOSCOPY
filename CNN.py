#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.optimizers import SGD
import pickle
import time
import cv2

# NAME="Ulcers-cnn-{}".format(int(time.time()))


X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X / 255.0

dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]
y = np.array(y)


for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "Ulcers-cnn_new-{}-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()
            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer - 1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))
                model.add(Dropout(0.2))

            model.add(Dense(1))
            model.add(Activation('softmax'))

            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

            # OR CATEGORICAL
            model.compile(loss="categorical_crossentropy",
                          optimizer="adam",
                          metrics=['accuracy'])

            model.fit(X, y, batch_size=5, epochs=1, validation_split=0.3, callbacks=[tensorboard])

model.save("ulcer-CNN.model")
#print(model.summary())

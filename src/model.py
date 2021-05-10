
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import numpy as np
import cv2
import os
from data import x_train, x_test, y_train, y_test

# create model
model = Sequential()

# add model
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',
          input_shape=(28, 28, 1), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, validation_data=(
    x_test, y_test), epochs=40, batch_size=100)

model.save('model2.h5')

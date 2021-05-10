from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import numpy as np
import cv2
import os
from data import x_train, x_test, y_train, y_test

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, validation_data=(
    x_test, y_test), epochs=40, batch_size=100)

model.save('model_fc.h5')

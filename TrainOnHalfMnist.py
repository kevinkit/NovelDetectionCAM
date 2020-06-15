# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:14:40 2020

@author: Kevin
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Dropout, Flatten, Conv2D,MaxPooling2D, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model
import numpy as np

#from  keras.datasets import cifar10

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()


# sort out some
y_train_known = y_train[np.squeeze(y_train) < 5]
y_train_novel = y_train[np.squeeze(y_train) >= 5]
x_train_known = x_train[np.squeeze(y_train) < 5] / 255
x_train_novel = x_train[np.squeeze(y_train) >= 5] / 255


y_train_known_cat = to_categorical(y_train_known)
y_test_cat = to_categorical(y_test)
img_inputs = keras.Input(shape=x_train[0].shape)

x = Conv2D(16,3,activation="relu")(img_inputs)
x = Conv2D(32,3,activation="relu")(x)
x = MaxPooling2D(3)(x)
x = Conv2D(32,3,activation="relu")(x)
x = Conv2D(16,3,activation="relu")(x)

x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(2048)(x)
x = Dropout(0.5)(x)
x = Dense(5,activation="softmax")(x)

model = Model(img_inputs,x)
model.summary()
model.compile("adam",loss="categorical_crossentropy",metrics=["acc"])
model.fit(x_train_known,y_train_known_cat,epochs=20)



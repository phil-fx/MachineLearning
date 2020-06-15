import os 

import matplotlib.pyplot as plt 
import numpy as np 

import tensorflow as tf 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *

from plotting import *


# MNIST-Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Cast to float32 für Numpy
x_train = x_train.astype(np.float32)  
y_train = y_train.astype(np.float32)  
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# Dataset varibles
train_size = x_train.shape[0]   # 60'000
test_size = x_test.shape[0]     # 10'000
num_timesteps = 784  # oder 28 - zeilenmäßiges einspeisen
num_features = 1     # und 28   # 28x28 px = 784, es muss zum geflatteten image werden
num_classes = 10                # 10 Zahlen

# Compute the categorical classes (OneHot-like)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Reshape image wird zu 60'000 x 784 (input data)
x_train = x_train.reshape(train_size, num_timesteps, num_features) # 60'000 Image-Daten werden geflatte von 28x28 --zu--> 784x1
x_test = x_test.reshape(test_size, num_timesteps, num_features)    # print('Geflatteter Shape von x_train & x_test: ',x_train.shape, x_test.shape) # 60'000 x 784 & 10'000 x 784

# Model params
lr = 0.001
optimizer = Adam(lr=lr)
epochs = 3
batch_size = 256 # Trade-Off Geschwind. // Performance ~2er-Potenz~ [32-1024]
units = 50
return_sequences = False

# Define the DNN 
model = Sequential()
# LSTM odeer RNN
model.add(LSTM(units=units, return_sequences=return_sequences, input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Dense(units=num_classes))
model.add(Activation('softmax'))
model.summary()

# Compile and train (fit) the model, afterwards evaluate the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])

model.fit(
    x_train, 
    y_train, 
    epochs=epochs,
    batch_size=batch_size,
    validation_data=[x_test, y_test])

score = model.evaluate(
    x_test, 
    y_test,
    verbose=0)
print('Score: ', score)

import os 
import time

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
from tensorflow.keras.callbacks import *

from plotting import *

# MNIST-Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Cast to float32 für Numpy
x_train = x_train.astype(np.float32)  
y_train = y_train.astype(np.float32)  
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# Reshape images to depth dimension
x_train = np.expand_dims(x_train, axis=-1)  # 1 dim mehr für die depth info
x_test = np.expand_dims(x_test, axis=-1)    

# Dataset varibles
train_size = x_train.shape[0]   # 60'000
test_size = x_test.shape[0]     # 10'000
width, height, depth = x_train.shape[1:] # ab stelle 1 (shape ist 60000, 28, 28, 1)
num_features = width * height * depth  # 28x28 px = 784, es muss zum geflatteten image werden
num_classes = 10                # 10 Zahlen

# Compute the categorical classes (OneHot-like)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

modelID = str('mnist_CNN_modelAPI_model_')
zeit = str(time.time())

# Save Path 
dir_path = os.path.abspath('/home/phil/MachineLearning/models/')
if not os.path.exists(dir_path):
    os.makdir(dir_path)
mnist_model_path = os.path.join(dir_path, str(modelID) + str(zeit) + '.h5')

# Log Dir 
log_dir = os.path.abspath('/home/phil/MachineLearning/logs/')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
model_log_dir = os.path.join(log_dir, str(modelID) + str(zeit))

# Model params
lr = 0.001
optimizer = Adam(lr=lr)
epochs = 5
batch_size = 256 # Trade-Off Geschwind. // Performance ~2er-Potenz~ [32-1024]

# Define the DNN 
input_img = Input(shape=x_train.shape[1:])

x = Conv2D(filters=4, kernel_size=3, padding='same')(input_img)
x = Activation('relu')(x)
x = Conv2D(filters=4, kernel_size=3, padding='same')(x)
x = Activation('relu')(x)
x = MaxPool2D()(x)

x = Conv2D(filters=8, kernel_size=5, padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(filters=8, kernel_size=5, padding='same')(x)
x = Activation('relu')(x)
x = MaxPool2D()(x)

x = Flatten()(x)
x = Dense(units=64)(x)
x = Activation('relu')(x)
x = Dense(units=num_classes)(x)

y_pred = Activation('softmax')(x)

# Build the model
model = Model(inputs=[input_img], outputs=[y_pred])

model.summary()

# Compile and train (fit) the model, afterwards evaluate the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])

tb = TensorBoard(
    log_dir=model_log_dir,
    histogram_freq=1,
    write_graph=True)

model.fit(
    x_train, 
    y_train, 
    epochs=epochs,
    batch_size=batch_size,
    validation_data=[x_test, y_test],
    callbacks=[tb])

score = model.evaluate(
    x_test, 
    y_test,
    verbose=0)
print('Score: ', score)

model.save_weights(filepath=mnist_model_path)
model.load_weights(filepath=mnist_model_path)

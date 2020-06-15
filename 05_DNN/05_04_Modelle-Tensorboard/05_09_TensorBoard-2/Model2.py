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
from tensorflow.keras.callbacks import *

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
num_features = 784              # 28x28 px = 784, es muss zum geflatteten image werden
num_classes = 10                # 10 Zahlen

# Compute the categorical classes (OneHot-like)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Reshape image wird zu 60'000 x 784 (input data)
x_train = x_train.reshape(train_size, num_features) # 60'000 Image-Daten werden geflatte von 28x28 --zu--> 784x1
x_test = x_test.reshape(test_size, num_features)    # print('Geflatteter Shape von x_train & x_test: ',x_train.shape, x_test.shape) # 60'000 x 784 & 10'000 x 784

# Save Path 
dir_path = os.path.abspath('/home/phil/MachineLearning/models/')
if not os.path.exists(dir_path):
    os.makdir(dir_path)
mnist_model_path = os.path.join(dir_path, 'mnist_model_phil.h5')

# Log Dir 
log_dir = os.path.abspath('/home/phil/MachineLearning/logs/')
if not os.path.exists(dir_path):
    os.makdir(dir_path)

log_dir_model2 = os.path.join(log_dir, 'model2')
if not os.path.exists(log_dir_model2):
    os.mkdir(log_dir_model2)

# Model params
init_w = TruncatedNormal(mean=0.0, stddev=0.01)
init_b = Constant(value=0.0)
lr = 0.001
optimizer = Adam(lr=lr)
epochs = 20
batch_size = 256 # Trade-Off Geschwind. // Performance ~2er-Potenz~ [32-1024]

# Define the DNN 
model = Sequential()

model.add(Dense(units=100, kernel_initializer=init_w, bias_initializer=init_b, input_shape=(num_features,)))
model.add(Activation('relu'))

model.add(Dense(units=80, kernel_initializer=init_w, bias_initializer=init_b))
model.add(Activation('relu'))

model.add(Dense(units=num_classes, kernel_initializer=init_w, bias_initializer=init_b))
model.add(Activation('softmax'))
model.summary()

# Compile and train (fit) the model, afterwards evaluate the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])

tb = TensorBoard(
    log_dir=log_dir_model2,
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



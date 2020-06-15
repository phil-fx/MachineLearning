import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *

np.random.seed(42)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image = x_train[0]
image = image.reshape((28, 28))

kernel = np.random.uniform(low=0.0, high=1.0, size=(2,2))
#kernel = np.array([[0, 0.5], [0.1, 0.0]])

# Stride (1,1)
# Conv Funktion definieren und anschließend plotten

def conv2D(image, kernel):
    shape = ((image.shape[0]-2), image.shape[1]-2)
    img = np.ones(shape)
    for i in range(image.shape[0]-2):
        for j in range(image.shape[1]-2):
            box = np.array([[image[i, j], image[i+1, j]],[image[i+1, j], image[i+1, j+1]]], dtype=np.uint8)
            Cprod = np.array((box * kernel), dtype=np.uint8)
            Cpix = np.sum(Cprod)
            img[i, j] = Cpix
    return img

conv_image = conv2D(image, kernel)

# Input und Outputbild des Pooling Layers mit imshow() ausgeben
plt.imshow(image, cmap="gray")
plt.show()

plt.imshow(conv_image, cmap="gray")
plt.show()
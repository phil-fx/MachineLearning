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

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image = x_train[0]
image = image.reshape((28, 28))
pool = (2, 2)

# Max-Pooling Funktion definieren und auf ein Bild aus dem
# MNIST Dataset anwenden.
# 2x2, max

def max_pooling(image, pool):
    cols, rows = image.shape #28x28
    p_size_col, p_size_row = pool #(2, 2)
    p_img_col_s = int(cols/p_size_col)
    p_img_row_s = int(rows/p_size_row)
    p_img_shape = (p_img_col_s, p_img_row_s) #14x14
    img = np.zeros(shape=(p_img_shape), dtype=np.float32) #14x14
    
    for i in range(p_img_col_s):
        for j in range(p_img_row_s):
            w = image[p_size_col*i:p_size_col*i+p_size_row, p_size_row*j:p_size_row*j+p_size_col]
            print(w)
            px_max = np.max(w)
            img[i, j] = px_max
    return img

pooling_image = max_pooling(image, pool)

print(image.shape)
print(pooling_image.shape)

# Input und Outputbild des Pooling Layers mit imshow() ausgeben
plt.imshow(image, cmap="gray")
plt.show()

plt.imshow(pooling_image, cmap="gray")
plt.show()
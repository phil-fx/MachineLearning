import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

from helper import *

class Model:
    def __init__(self):
        self.x = tf.Variable(tf.random.uniform(shape=[2], minval=-2.0, maxval=2.0)) # x= [x0, x1]
        self.learning_rate = 0.001
        self.optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate) # Stochastic Gradient Descent, der optimizer bildet den Grad selbstständig
        self.current_loss_val = self.loss()

    def loss(self):
        self.current_loss_val =  100 * (self.x[0]**2 - self.x[1])**2 + (self.x[0] - 1)**2
        return self.current_loss_val

    def fit(self):
        self.optimizer.minimize(self.loss, self.x) # loss function, variablen welche angepasst werden
        
model = Model()
downhill_points = []

for it in range(5000):
    model.fit()
    if it % 100 == 0:
        print(model.x.numpy(), model.current_loss_val.numpy())
        downhill_points.append(model.x.numpy())

plot_rosenbrock(downhill=downhill_points, x0=downhill_points[-1][0], x1=downhill_points[-1][1])
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error

from helper import *

def mae(y_true, y_pred):
    error = np.mean(np.abs(y_true - y_pred))
    return error

def mse(y_pred, y_true):
    error = np.mean((y_true - y_pred)**2)
    return error

x, y = regression_data()
x = x.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

regr = LinearRegression()
regr.fit(x_train, y_train) # Training
print(regr.score(x_test, y_test)) # Testing
y_pred = regr.predict(x_test)

MAE = mean_absolute_error(y_test, y_pred)
print('MAE: ', MAE)
MSE = mean_squared_error(y_test, y_pred)
print('MSE: ', MSE)
print('selbstgeschrieben...')
print('MAE: ', mae(y_test, y_pred))
print('MSE: ', mse(y_test, y_pred))

plt.scatter(x, y)
plt.plot(x_test, y_pred)
plt.show()
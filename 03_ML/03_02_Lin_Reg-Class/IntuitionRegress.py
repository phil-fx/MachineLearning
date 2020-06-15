import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

from helper import regression_data

x, y = regression_data()

# plt.plot([x for x in range(-15, 15, 1)], [2*x+5 for x in range(-15, 15, 1)], c='red')
m = 2.0
b = 5.0
y_pred = [m*xi + b for xi in x]

plt.scatter(x, y)
plt.plot(x, y_pred, c='red')
plt.show()

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

from helper import classificiation_data

x, y = classificiation_data()

m = -4.0
b = 6.5
border = [m*xi + b for xi in x]

colors = np.array(['red', 'blue'])
plt.scatter(x[:,0], x[:,1], c=colors[y[:]])
plt.plot(x, border)
plt.show()
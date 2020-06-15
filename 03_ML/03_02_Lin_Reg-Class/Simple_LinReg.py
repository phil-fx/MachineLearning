import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

from helper import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x, y = regression_data()
print(x.shape)
#print(x)
x = x.reshape(-1, 1) # muss gemacht werden da sklearn und keras jedes elemet als separte Unterliste erwartet
print(x.shape)
#print(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

regr = LinearRegression()
regr.fit(x_train, y_train) # Training
print('Der R2-Score beträgt: ', regr.score(x_test, y_test)) # Testing
y_pred = regr.predict(x_test)

# m = 2.0
# b = 5.0
# y_pred = [m*xi + b for xi in x]

plt.scatter(x, y)
plt.plot(x_test, y_pred, c='red')
plt.show()

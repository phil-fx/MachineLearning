import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

def f(x):
    return x**4 + 5*x**3 + 14*x**2 + x + 10

x = np.linspace(start=-10.0, stop=10.0, num=1000).reshape(-1, 1)
y = f(x)

def relu(x):
    if x > 0: return x
    else: return 0

#Linear Regression
model = Sequential()             # 
model.add(Dense(500))             # Stelle 0: Input --> Relu
model.add(Dense(1))              # Stelle 2: Hidden --> output
model.compile(optimizer=Adam(lr=5e-2), loss='mse')
model.fit(x, y, epochs=20)
y_pred_linear = model.predict(x)

#Relu
model = Sequential()              # 
model.add(Dense(200))             # Stelle 0: Input --> Relu
model.add(Activation('relu'))     # Stelle 1: Relu --> Hidden0
model.add(Dense(200))             # Stelle 2: Hidden0 --> Relu
model.add(Activation('relu'))     # Stelle 3: Relu --> Hidden1
model.add(Dense(1))               # Stelle 4: Hidden1 --> output
model.compile(optimizer=Adam(lr=5e-2), loss='mse')
model.fit(x, y, epochs=20)
y_pred_relu = model.predict(x)

model = Sequential()                # 
model.add(Dense(100))               # Stelle 0: Input --> Relu
#model.add(Activation('sigmoid'))    # Stelle 1: Sigmoid --> Hidden0
#model.add(Dense(500))               # Stelle 2: Hidden0 --> Sigmoid
model.add(Activation('sigmoid'))    # Stelle 3: sigmoid --> Hidden1
model.add(Dense(1))                 # Stelle 4: Hidden1 --> output
model.compile(optimizer=Adam(lr=5e-2), loss='mse')
model.fit(x, y, epochs=20)
y_pred_sigmoid = model.predict(x)

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(24,12))
plt.title('linear reg // relu // sigmoid')
plt.grid(True)
ax1.plot(x, y, color='blue')
ax1.plot(x.flatten(), y_pred_linear.flatten(), color='red')
ax2.plot(x, y, color='blue')
ax2.plot(x.flatten(), y_pred_relu.flatten(), color='red')
ax3.plot(x, y, color='blue')
ax3.plot(x.flatten(), y_pred_sigmoid.flatten(), color='red')
plt.show()
plt.close()
import numpy as np
import matplotlib.pyplot as plt

def e_function(a, b):
    x = []
    y = []
    for i in range(a, b+1, 1):
        Wert = np.exp(i)
        y.append(Wert)
        x.append(i)
    return x, y

a = 2
b = 10
x, y = e_function(a,b)

plt.plot(x, y, color='red')
plt.legend('e-funktion')
plt.xlabel('x-Wert')
plt.ylabel('e-Fkt')
plt.show()

import numpy as np 
import matplotlib.pyplot as plt 

spieler = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# zufallszahl uniform
zahlen = np.random.randint(1, 11, 5)
print(zahlen)

# zufalls-elemente
gewinner = np.random.choice(spieler, 5)
print(gewinner)

# zufalls-reihenfolge
spieler_perm = np.random.permutation(spieler)
print(spieler_perm)

# normalverteile zufallszahlen
zahlen = np.random.normal(loc=5.0, scale=1, size=10)
print(zahlen)

# standardnormalverteilng zw 0 und 1
zahlen = np.random.randn(5)
print(zahlen)

x1 = np.random.multivariate_normal(mean=[5.0, 5.0], cov=np.identity(2), size=15)
print('Mehrdim Verteilung: ', x1)

x2 = np.ones(shape=(3, 2))
print('Einsmatrix : \n', x2)

x3 = np.identity(3)
print('Einsmatrix : \n', x3)

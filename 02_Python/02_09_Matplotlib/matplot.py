import matplotlib.pyplot as plt
noten = [10,20,30,50,80,120,250,540]
noten2 = [1,5,20,40,70,100,240,540]
plt.plot(range(8), noten, color='red')
plt.plot(range(8), noten2, color='blue')
plt.legend(['Erster', 'Zweiter'])
plt.xlabel('x-Achse')
plt.ylabel('y-Achse')
plt.title('Grafik')
plt.show()

# x = [4, 8, 2, 10]
# y = [5, 5, 2, 10]
# plt.scatter(x, y, color='green')
# plt.show()
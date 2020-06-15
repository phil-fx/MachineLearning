import numpy as np

my_list = []
my_list.append(12)
my_list.append(23)

for i in range(10):
    my_list.append(i)


my_list_comp = [i**2 for i in range (10) if i % 2 == 0]
my_list_comp2 = [i for i in range (10) if i % 2 == 0]
print(my_list_comp)
print(my_list_comp2)

m = np.array([[1, 0, 0, 1, 2, 3, 4, 5]])
print(m.shape)
m = np.reshape(m, (2, 4))
print(m)
print(m.shape[0])


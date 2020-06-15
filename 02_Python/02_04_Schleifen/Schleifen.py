for i in range(5):
    print('Das ist der ' , i, ' te durchlauf')

print('\n')

for i in range(1, 10, 2):
    print('Das ist der ' , i, ' te durchlauf')

print('\n')

for i in range(1, 10, 2):
    print('Das ist der ' , i, ' te durchlauf')

print('\n')

bin_pleite = False
kontostand = 10

while bin_pleite == False:
    print('Bin nicht pleite', kontostand)
    kontostand -= 1
    if kontostand <= 0:
        bin_pleite = True
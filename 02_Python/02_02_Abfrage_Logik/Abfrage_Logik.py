bin_pleite = None
bin_reich = None

kontostand = 4530.6

if kontostand > 0:
    bin_pleite = False

    if kontostand > 1000:
        bin_reich = True
    else:
        bin_reich = False
elif kontostand == 0:
    print ('Kontostand ist Null :(')
else:
    bin_pleite = True
    bin_reich = False

print('Bin ich pleite ?: ', bin_pleite)
print('Bin ich reich ?: ', bin_reich)
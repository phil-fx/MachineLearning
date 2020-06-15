noten_dict = {'Armin': 1, 'Ben': 2, 'Jan': 1}
armins_note = noten_dict['Armin']
print('Armins Note :', armins_note)

print('\n')

for schueler, note in noten_dict.items():
    print('Der ', schueler, 'hat die Note: ', note)

print('\n')

for schueler in noten_dict.keys():
    print('Der Schüler hat den Namen: ' ,schueler)
    print('Der Schüler hat den Namen: ' ,schueler , 'und hat die Note: ', noten_dict[schueler])

print('\n')

for note in noten_dict.values():
    print('Es gibt die Noten: ', note)
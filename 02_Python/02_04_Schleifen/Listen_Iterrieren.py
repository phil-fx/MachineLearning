noten = [1, 2, 3, 4, 1, 2]
for note in noten:
    print(note)

print('\n')

for i in range(len(noten)):
    print(noten[i])

print('\n')

faecher = ['Mathe', 'Deutsch', 'Geo', 'Sport', 'Englisch']
for note, fach in zip(noten,faecher):
    print('In ', fach, ' habe ich die Note: ' , note)

print('\n')

praeferenz = ['Mathe', 'Deutsch', 'Geo', 'Sport', 'Englisch']
for index, fach in enumerate(praeferenz):
    print('Das Fach :', fach , ' ist an Stelle', index)
from students import bester_schueler as bs

noten_dict = {'Armin': 10, 'Ben': 22, 'Jan': 100, 'Udo': 31}

name, note = bs(noten_dict)
print('Der beste Schüler ', name, 'hat die Punktzahl ', note)


#import students ## als Datei
#name, note = students.bester_schueler(noten_dict)


import random
zahl = random.randint(1,10)
print(zahl)
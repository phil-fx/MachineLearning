noten = [1, 2, 4, 5, 1, 1]
print(noten)
print('Der zweite Eintrag ist: ' , noten[1])

noten.append(6)

print('Der letzte Eintrag ist: ' , noten[-1])

noten.pop()
print('Jewtzt ist Der letzte Eintrag ist: ' , noten[-1])

noten.insert(0, 7)
print(noten)
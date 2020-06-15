noten_dict = {'Armin': 10, 'Ben': 22, 'Jan': 100, 'Udo': 31}

def bester_schueler(noten_dict, konsole=False):
    akt_bester_schueler = ''
    akt_beste_note = 0

    for name, note in noten_dict.items():
        if note > akt_beste_note:
            akt_beste_note = note
            akt_bester_schueler = name
        if konsole == True:
            print('Ausgabe :', akt_bester_schueler, '-->', akt_beste_note)
    return akt_bester_schueler, akt_beste_note

Vorname, Punkte = bester_schueler(noten_dict, True)
print('Der Schüler ' , Vorname, ' hat die beste Note, und diese lautet: ' ,Punkte)
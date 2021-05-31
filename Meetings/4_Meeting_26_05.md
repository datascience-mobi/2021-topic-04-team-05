# Meeting Notes
**26.05.2021**

---

## Last week's progress
- Report angefangen
- Dice Score fast fertiggestellt
- Plan bezüglich Synthetic Images aufgestellt und erste Synthetic images erzeugt

## Questions
- Warum können manche Packages nicht installiert werden?
A: conda install -c conda-forge opencv
- Wie Repo organisieren?
A: Jeder erstellt eigene Branch mit Python file und am Ende machen wir code review und eine der Branches wird mit main brainch gemergt

## Additional Notes

### Repo Struktur
Dice score Dateien mergen!
Repostruktur wie Beispielrepo
Funktion in Repo für Dice score → importieren in Report

### Report: 
in jupyter notebook!
-#%% md (für markdown chunk)
-#%% (für python chunk)
import 
working_on.dice_score.code.dice_score as dice score
für code: ``` nicht ‘’ (richtiges neben del taste)
für code der wie python gehighlited wird: ```python
Prinzip: Codedokumente erstellen & dann in Report importieren (nicht copy paste!)

### Git:
markdown & python
feature branch, develop branch, pull request nachdem der feature fertig gecoded ist + code review, 
man könnte nachher eine branch erstellen mit dem commit, den man zurückstellen will (oder copy paste:))

## Plans for next week
- Dice Score final fertigstellen
- Report-Teil zu Dice Score (und Synthetic images) schreiben
- Support Vector Machine Implementation beginnen

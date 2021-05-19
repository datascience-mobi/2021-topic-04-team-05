# Meeting Notes
**19.05.2021**

---

## Last week's progress
- Project Proposal
- Dice Score recherchiert und erste Schritte implementiert
- Synthetic images recherchiert und erste Schritte implementiert

## Questions
- Ground truth mit Grauwerten? → Anpassen mit Threshold damit nur weiß & schwarz?
  A: jeder Zellkern hat eigenen Grauwert -> so können Nuclei voneinander unterschieden werden
- was soll output unserer SVM sein? schwarz/weiß oder ränder
- Welchen Mehrwert haben synthetic images an DIESEM Schritt? (eigentlich ja benutzt um Dataset zu vergrößern); wie können wir am geschicktesten unseren Dice score damit überprüfen (ggf. Paper raussuchen)?
  -> für dice score reicht einen numpy array zu erzeugen
- Domain transfer/randomization -> sollen wir ein Programm wie Blender oder Unity verwenden oder eher einfach Lichtverhälltnisse durch Gradienten etc verändern?
  -> nicht die Programme, sondern eher mit Gradienten etc. arbeiten, also mit eigenem Code
- Git-Probleme Ria
- inwiefern muss unser Code neu sein?...wurde ja alle schon tausend mal gemacht

## Plans for next week
- switch topics
- implement Dice Score and Synthetic images code finally

# Meeting Notes
**02.06.2021**

---

## Last week's progress
- Report zu Dice Score und erster Teil zu Synthetic images
- Dice Score fertig + unit test -> erfolgreich (einziges Problem: Shape der Synthetic images)
- SVM implementation begonnen

## Questions
1. How were the ground truth images generated?
   A:As a ground truth image, researchers mostly use the segmentation result of humans. We will use the ground truth images provided with our data sets, which we suspect to be acquired by this method. 
2. Git: already OK? and what about_init_.py (for what)
#init.py (leer) converts dicescore.py to module
#modules can be imported

3. How to import the code from dicescore.py in our report? 
````
import sys
sys.path.append(C:\Users\glori\PycharmProjects\2021-topic-04-team-05\Finalmodules)
from Finalmodules import dicescore.py
````
#brauchen wir die ``?
#wie kann der code importiert werden?
%cd Finalmodules
%load dicescore.py



## Plans for next week
- Continue with SVM
- Report fir SVM
- Solve problems with synthetic images 

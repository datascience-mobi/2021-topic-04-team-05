# Meeting Notes
**26.05.2021**

---

## Last week's progress
- We already started with the report, had plans about the synthetic images, and generated the first synthetic images
- Dice Score is ready

## Questions
- Why couldn't some packages be installed?
  A: conda install -c conda-forge opencv
- How can we organize our repository?
  A: Every one of us own a branch with Python file and at the end we review our codes. One 'best'  branch would be merged with the master.

### Repository structure
Merge the dice score data
Repository structure like the example
Import the function in repository for the dice score → Report

### Report: 
In Jupyter Notebook!
-#%% md (for markdown chunk)
-#%% (for python chunk)
import 
working_on.dice_score.code.dice_score as dice score
für code: ``` nicht ‘’ (richtiges neben del taste)
für code der wie python gehighlited wird: ```python
Principle: generate code documents and then import in Report (no copy paste!)

### Git:
Markdown & Python
Feature branch, develop branch, pull request according the status of the features + code review
We could generate one branch with the commit that can be reset?

## Plans for next week
- Finish the final dice score cod
- Write part of the report for dice score and synthetic images
- Start the Support Vector Machine Implementation 

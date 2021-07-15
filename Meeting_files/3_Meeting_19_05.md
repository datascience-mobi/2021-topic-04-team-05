# Meeting Notes
**19.05.2021**

---

## Last week's progress
- Project Proposal
- More research about the dice score and implement the first step
- Research about synthetic images and implement the first step

## Questions
- Ground Truth with grayscale? Should it be adapted with Threshold Function so that the images'd be black and white?
  A: Each nucleus has its own gray value so that it can be differentiated from the background and other nuclei
- What should our SVM output look like? black and white or just with edges
- Which advantages do synthetic images have in THIS step? How can we use these synthetic images to further train our dice score with
  the most elegant way? (maybe research papers)
  A: It's enough for the dice score to generate numpy array
- Domain transfer/randomization -> should we use a program like Blender or Unity, or should be just adjust the light propotion with gradient?
  A: not the programm, but with gradient and own code
- Git-Problems from Ria
- To what extent should the code be new? So many similar SVM codes were already made and implemented

## Plans for next week
- Switch topics
- Implement Dice Score and Synthetic images code finally

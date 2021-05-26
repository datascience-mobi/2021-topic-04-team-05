# Report of Project 5: 'Implementation and evaluation of machine learning (support vector machine) segmentation
## Data analysis project for bachelor in molecular biotechnology at Heidelberg University
### 19.07.21
### Authors: Michelle Emmert, Juan Hamdan, Laura Sanchis and Gloria Timm

*ich werde kommentare, anmerkungen, noch zu klärende fragen... immer in kursiv schreiben* 

*Idee: unseren algorithmus anhand eines bildes erklären und dieses immer wieder im report zeigen um die veränderungen 
zu zeigen & am ende dice berechnen --> veranschaulichen*


# Abstract





# Table of contents *ggf. noch 2.1, 2.2 etc.*
####1. Introduction
####2. Our Datasets
####3. 
####4. Pre-processing
####5. Implementation of support vector machine
####6. Evaluation using the Dice coefficient
####7. Results
####8. Discussion
####9. Bibliography





# 6. Dice coefficient

The Dice coefficient is a score to evalutate and compare the accuracy of a segmentation (method).
Needed for its calculation are the segmented image as well as a corresponding binary reference point also called 
ground truth.
As ground truth image researchers mostly use the segmentation result of humans. We will use the ground truth images 
provided with our data sets, which we suspect to be aquired by this method.
Using the ground truth image it is possible to assign the labels true positive (TP), false positive (FP) and false 
negative (FN) to each pixel of the segmented image. 


# Synthetic images
## Mixing object over real world scenes
--> probably not as useful for our case as cells are usually in front of dark background, but still an option to evaluate --> will the Dice Score get better with that method?

## domain transfer/randomization
as size, lighting etc. of objects change, this can be simulated with domain transfer (Unity or Blender)
--> probably more useful, as we have changes in size, dividing or leaving cells etc


# Support Vector Machine

1. Tensorflow --> an Option?




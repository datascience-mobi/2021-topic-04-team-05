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
## 6.1 Calculating the dice coefficient

The Dice coefficient is a score to evaluate and compare the accuracy of a segmentation (method).
Needed for its calculation are the segmented image as well as a corresponding binary reference point also called 
ground truth.
Image researchers mostly use as ground truth the segmentation result of humans. We will use the ground truth images 
provided with our data sets, which we suspect to be acquired by this method.
Using the ground truth image, it is possible to assign the labels true positive (TP), false positive (FP) and false 
negative (FN) to each pixel of the segmented image.
This information is then used to calculate the dice coefficient using formula (1):

(1) dice = ${\frac{2TP}{2TP + FP + FN}}$ 
*ich habe keine ahnung wie ich das als formel implementieren kann*

The dice is element of [0,1]. 0 indicates that the ground truth and the segmentation result do not overlap. 1 on the 
other hand shows a 100% overlap of ground truth and segmented image.

## 6.2 Implementing the dice coefficient
*ich weiß nicht warum das nicht als code angezeigt wird* 

'''
###import images (prediction & ground truth) as arrays

###compute dice score
```python
def dice_coefficient(imgt, imgp):  # t = ground truth, p = SVM prediction
    assert imgt.dtype == np.bool #the images with type array are converted to type bool
    assert imgp.dtype == np.bool #the images with type array are converted to type bool
    intersection = np.logical_and(imgt, imgp) #compute the truth value of x1 AND x2 element-wise = sums all pixels where both gt and pred have the value 'true'
    union = imgt.sum() + imgp.sum() #compute the truth value of x1 OR x2 element-wise = sums all pixels where either gt or pred (or both) have the value 'true'
    if intersection + union == 0:
        return ('dice cannot be calculated - no intersection') #because it is mathematically not allowed to divide by 0, which would happen if gt and pred don't intersect
    else:
        dice = (2 * intersection) / (union + intersection) #using the dice formula to calculate the dice IF gt and pred intersect
        return dice #print out dice
```


## 6.3 Synthetic images
## How does it work, and what is our goal?
The concept behind creating synthetic images is to use algorithms and images which are already available to generate new ones. 
Although our first objective was to just use these new images to test our code for the dice 
score, we realized while researching for this topic that synthetic images have an immense 
potential, most of all for the training of machine learning algorithms. The bigger the training 
dataset is, the better the performance of the program. Our data set consists of XXXXX 
images of cell nuclei, and because we have to slit it up between the training and the 
test data set, we won't be able to use all images to train our SVM. Because of this,
we decided to implement 
our synthetically produced images not only for the testing of the dice score, but to enlarge our 
training data pool for our Support Vector Machine, and check afterwards if its efficiency is 
better with our dice score. 
There are many methods that can be used in order to generate synthetic images. Because of the 
scope of our project and the kind of images that we want to produce, we focused on image 
composition and domain transfer.

## Image composition
Image composition consists of taking various foreground images, which have been segmented out of 
their backgrounds or have a .png format to begin with, and paste them onto different backgrounds.
The foreground images can be modified by using different light conditions, contrasts, zooms or 
rotations in order to achieve more variety in the results.
--> probably not as useful for our case as cells are usually in front of dark background, but 
still an option to evaluate --> will the Dice Score get better with that method?


*Here we could insert the code from my Jupyter notebook -> issue: it hasn't been written by us 
  (and it hasn't been modified either...), so maybe it would be better to just use our own code 
  when we write it.* 

## domain transfer/randomization
The idea here is that there is a model from the object class you want to train your model for, 
and in that model, every parameter from the object and its environment that is not necessary for 
its recognition by the machine has been randomized.
This means for example size, lighting or color, and there are very powerful tools to do this,  like 
Unity or Blender.
--> probably more useful, as we have changes in size, dividing or leaving cells etc, but also a 
bit of an overkill, as it is usually used to train robots to work from a simulation onto reality.

# Support Vector Machine

1. Tensorflow --> an Option?




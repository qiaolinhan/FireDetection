# video based early wildfire detection with fully convolutional network

## Lecture 1
Feifei Li, 
ImageNet-->billions of tons of image program
        -->CNN win in 2012
        -->Adaboost, AlexNet, 19 layers to 150 layers
        -->still many things need to solve, action motion detection.
        
## Lecture 2
Python or C++
Google clould if needded
Image classification, lable.-->hard for maachine.It sees that 
                               an image is just big grid of numbers between [0,255]  
* problem: Semantic Gap                               
* challenges: Veiwpoint variation, Illumination, Deformation, Occlusion （only a face）, 
              Background Clutter (same color as background), 
              Intraclass variation (different shapes and ages)  
* API an image classifier in python
```
def classify_image (image):
  #
  return class_label
```
we need algorithms to scale much more naturally to all variety of objects.-->  
Data-Driven Approach
  1. collect a dataset of images and labels (collect on the internet, many and many)
  2. use machine learning to train classifier
  3. evaluate the classifier or new images
```
def train(image,labels):
  #machine learning!
  return model
```
```
def predict(model,test_images):
  #use model to predict labels
  return test_labels
```
function change from 1 to 2-->'this is cat' to train and prediction.  

#### Nearest Neighbor (NEVER USED <-- slow at test, distance calculation not very well)
  train: memmorize all data and labels  
  predict: predict the label of the most similar training image  
  example: example dataset: CIFAR10 for machine learing. 10 classes, 50000 training images, 10000 testing images.
  Q: How to compare images?
  A: L1 distance:  $ d_1(I_1,I_2)=\sum_{p}|I1^P-I_2^P| $ （install the maxjax and        see euqations）(sometimes also called Manhattan distance)
     Stupid, but sometimes it give us very concrete way to measure difference between two images.  
```python
# full nearest neighbor classifier
import numpy as np
class NearestNeighbor:
  def _init_(self):
    pass
  
  #memorize training data
  def train (self,X,y):
    '''X is N*D where each row is an example, Y is 1D of size N'''
    # the nearest neighbor classifier simply remember all training data
    self.Xtr=X
    selfytr=y
  def predict(self,X):
    '''X is N*D, each row is an example we wish to predict label for'''
    num_test=X.shape[0]
    #make sure that the output type matches the input type
    Ypred=np.zeros(num_test,dtype=self.ytr.dtype)
    
    #For each test image, find closest train image and predict label of nearest image
    #loop over all test rows
    for i in Xrange(num_test):
    #find the nearest training image to the ith test image
    #using the L1 distance (sum of absolute value differences)
    distance = np.sum(np.abs(self.Xtr-X[i,:]),axis=1)
    min_index=np.argmin(distance) #predict the label of the nearest example
  return Ypred
```
  Q: With N examples, how fast the trainning and prediction could be?  
  A: Train O(1), predict O(N)  
     BAD, we want classifier FAST at prediction; SLOW for training is okay.  not good on mobile.
  * K-Nearest Neighbor, instead of copying from nearest neighbor, take majority vote from K closest points.
  K-Nearst Neighbors use DISTANCE METRIC to compare images (L2)  
  L1 (Manhattan) distance--square $ d_1(I_1,I_2)=\sum_{p}|I_1^p-I_2^p| $  
  L2 (Euclidean) distance --circle $ d_2(I_1,I_2)=\sqrt{\sum_{p}(I_1^p-I_2^p)^2} $  
  * L1 distance depends on the choice of coordinates system
  * L2 distance doesn't matter with different coordinates.  
  Q: The best value of K and the best distance to use?
  A: These arer hyperparameters: choices about the algorithm that we set rather than learn. `Ther is not way to learn them directly from the data.`  
  IMPORTANT: seperate the validation data and the test data.
  * Cross Validation--> stratagy for setting hyperparameters, get more robust hyperparameters.  
  Eg: 5-folds cross validation for value of K in K-nearest heighbor classifier. _Seems that K~=7 works best)
  
* When working with machine learing, end with the plot that contains the information of `ACCURACY`,`PERFORMANCE OF YOUR HYPERPARAMETERS`.

* Curse of dimentionality
  Dimentions=1, Points=4
  Dimention=2, Points=$4^2$
  Dimention=3, Points=$4^3$
  
* So, for the K-Nearest Neighbors:
  In image classification, start with trainig set of iages and lables and must predict labels on the test set.  
  K-Nearest Neighbors classifier predicts labels based on the nearest trainig examples.  
  Distance metric and K are hyperparameters.  
  Choose hyperparameters using the validation set; only run on test set ONCE at the very end!  

#### Linear Classification
Linear classifier  
CIFAR10: 50000 training images 32*32*3 (3 color channels)
Parametric model, two different components  
image-->_f(x,W)_-->10 numbers giving class scores.  
W: parameters or weights.  

* Deeplearning-->the right structure of function F, the simplest way is to multiply W and x. _f(x,W)_=W*x.  

Eg: image with 4 pixels and 3 classes  
Problem: The linear classfier is just learning tamplate for each class
* Loss function--> Quantifying wht it means to have good W.
* Optimizatin--> start with random W and find W that minimizes the loss.
* ConvNets --> Tweak the functional form of _f_.
 

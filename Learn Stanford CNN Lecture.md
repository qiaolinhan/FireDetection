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

* Nearest Neighbor
  train: memmorize all data and labels  
  predict: predict the label of the most similar training image  
  example: example dataset: CIFAR10 for machine learing. 10 classes, 50000 training images, 10000 testing images.
  Q: How to compare images?
  A: L1 distance: 
  ```
  $ d_1(I_1,I_2)=\sum_{p}|I1^P-I_2^P| $
  ```
  
    
  


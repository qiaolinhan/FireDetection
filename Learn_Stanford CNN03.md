
## Lecture 3
##### RECALL
Loss functions, Optimization  
Challenges of recognition: Viewpoint, Illumination, Deformation, Occlision, Clutter, Intraclass Variation.  
Data-driven approach, K-Nearest Neighbors (KNN).  
Linear Classifier $ f(x,W)=Wx+b $ Outputs the scores.  
  * How to actually chosse the W?
  1. Define a loss function that quatifies our unhappiness with the score across the training data.  
  2. Come up with a way of efficiently finding the parameters that minimize the loss function. (*Optimization*)  


#### Loss Function and Regularization
$ f(x,W)=Wx $  
A *Loss function* tells how good our current classifier is  
Given a dataset of examples:  
$ {(x_i,y_i)}_{i=1}^N $  
Where $ x_i $ is image and $ y_i $ is (integer) $ lable   
Loss over the dataset is sum of loss over examples:
$ L=\frac{1}{n} \sum_{i}L_i(f(x_i,W),y_i) $  
* To concrete loss function, there is a multi-class SVM loss.  
It is a generalization of SVM to handle multiple classes.
Given an example $(x_i,y_i)$ where $ x_i $ is the image and where $ y_i $ is the lable (integer), and using the shorthand for the score vector: $ s=f(x_i,W) $  
The SVM loss has the form:
$$ L_i=\sum_{j\neq y_i} {0\ \if s_{y_i}\geq s_j+1 
                         s_j-s_{y_i}+1\ \otherwise}
      =\sum_{j\neq y_i}max（0，s_j-s_{y_i}+1) $$  
Where $ s $ is the  predict score, $ s_{y_i} $ is the score of true class.
* Hinge loss $ \uparrow $
* Only concern about the different scores 
 
* squared loss: mistake will be very very bad!
* hinge loss: we don't actually care between being a little wrong and being a lot wrong.
$ L_i=\sum_{j\neq y_i}max(0,s_j-s_{y_i}+1) $
```python
#nulticlass SVM Loss
import numpy
def L_i_vectorized(x,y,W):
  score=W.dot(x)
  margins=np.maximum(0,scores-scores[y]+1)
  margins[y]=0
  loss_i=np.np.sum(margins)
  return loss_i
```

$ f(x,W)=Wx $
$ L=\frac{1}{N} \sum_{i=1}^N \sum_{j\neq {y_i}} max(0,f(x_i;W)_j-f(x_i;W)_{y_i}+1) $

$ L(W)=\frac{1}{N}\sum_{i=1}^{N}L_i(f(x_i,W),y_i)+\lambda R(W) $  
* Data loss: Model predictions should match training data， use the training data to find some classifier. `We only care about the performance on test data`
* Regularization: Model should be "simple", so it works on test data.  
* Whole idea: Occam's Razor.  

$ \lambda $ =regularization strength (hyperparameter)
###### Common Use Regularization
* #L2 regularization#: $ R(W)=\sum_{k}\sum_{l}W_{k,l}^2 $
* L1 regularization: $ R(W)=\sum_{k}\sum_{l}|W_{k,l}| $ (encouraging sparity in the matrix W)
* Elastic net (L1+L2) $ R(W)=\sum_k\sum_l\beta W_{k,l}^2+|W_{k,,l}| $   (ML DL)
* Max norm regularization
* Dropout
* Fancier: Batch normalization, stochaastic depth

* Popular in DL: Softmax Classifier (MUltinomial Logistic Regression), there score have meaining: Unnormalized log probabilities of the class  
$ P(Y=k|X=x_i)=\frac{e^s_k}{\sum_je^{s_j}} $ where $ s=f(x_i;W) $  
$ \frac{{e^s}_k}{\sum_j e^{s_j}} $ is called softmax function. 
We Want to maximize the log liklihood, or (for a lossfunction) to minimmize the negative log likelihood of the correct class: $ L_i=-logP(Y=y_i|X=x_i) $  
* `loss funtion measures bad not good.`
unnormalized log probabilities $ \rightarrow (exp) $ unmormalzied Probabilities $ \rightarrow (normalize) $ probalilities $ $ \rightarrow $ $ L_i=-log(probabilities) $
* In summary $ L_i=-log(\frac{e^sy_i}{\sum_j {e^s}_j}) $  

##### Recap
- We have some dataset of (x,y)  
- We have a #socre function#: $ s=f(x;W)\overset{e.g.} Wx $  
- We have a #loss function#: 
  1. softmax: $ L_i=-log(\frac{e^sy_i}{\sum_{j} e^{s_j}) $  
  2. SVM: $ L_i=\sum_{j\neq y_i} \max (0,s_j-s_{y_i}+1) $  
  3. Full loss: $ L=\frac{1}{N} \sum_{i=1}^{N}L_i+R(W) $  

Q: How do we find this W that minimize the loss?
A: Optimization

#### Optimization
#1. Random search (bad idea solution)#
```python
import numpy as np
# assume X_train is the data where each column is an example
# assum Y_train are the labels
# assume the function L evaluates the loss function
bestloss=float ("inf") #python assigns the highest possible float value
for num in xrange(1000):
  W=np.random.randn(10,3073)*0.0001 #generate random parameters
  loss=L(X_train,Y_train,W) #get the loss over the entire training set
  if loss<bestloss:
    bestloss=loss
    bestW=W
  print 'in attempt %d the loss was %f, best %f' %(num,loss,bestloss) 
```
when test on the test set:
```
scores=Wbest.dot(Xte_cols) #10x1000, the class scores for all the test examples
#find the index with max score in each colum (the predicted class)
Yte_predict=np.argmax(score,axis=0)
# and calculate accuracy (fraction of predictions that are correct)
np.mean(Yte_predict==Yte)
#return 0.1555
```

#2. Follow the slope (can be used to train NN and others)#
In 1-dimention, the derivative of the function:
$ \frac{df(x)}{dx}=\lim\underset{h\rightarrow 0}\frac{f(x+h)-f(x)}{h} $  
In multiple dimentions, the gradient is the vector of (partial derivatives) along each dimention.  
The slope in any direction is the #dot product# of the direction with the gradient.  
The direction of steepest decent is the #negtive gradient#.
* Use calculus to compute an analytic gradient 

##### In summary
- Numerical gradient: approximate, slow, easy to write  
- Analytic gradient: exact, fast, error-prone

`To practice: Always use analytic gradient, but check implementation with numerical gradient. This is called a _gradient check_.` 

#### Gradient Decent
```python
#Vanilla Gradient Decent
while True:
  weights_grad=evaluate_gradient(loss_fun,data,weights)
  weights+=-step_size*weights_grad # perform parameter update
```
* Stochastic Gradient Decent (SGD)
$ L(W)=\frac{1}{N}\sum_{i=1}^{N}L_i(x_i,y_i,W)+\lambda R(W) $  
Full sum expensive when N is large! ---->super slow when N large!
$ \triangledown _W L(W)=\frac{1}{N}\sum_{i=1}{N}\triangledown _WL_i(x_i,y_i,W)+\lambda \triangledown _W R(W) $  
Approximate sum using a minibatch of example (32/64/128 common used)  
```python
while True:
   data_batch=sample_training_data(data,256)
   weights_grad=evalue_gradient(loss_fun,data_batch,weights)
   weights+=-step_size*weights_grad #perform parameter update
 ```
 Q: Why stochastic
 A: It uses small minibatch to compute an estimate of the full sum and an estimate of the true gradient, `it can be viewed as a Monte Carlo estimate of some expectation of the true value.`
 
##### Image Fratures
* Linear classifier: Maybe just taking raw image pixels and then feeding the raw pixelsthemselves
* Two stage approach: take image, compute various feature representations of the image.
###### Motivation
Cannot separate with linear classifier $ \rightarrow f(x,y)=(r(x,y),\theta(x,y)) $ After applying feature transform, points can be separated  
common feature vectors:
* Color Histogram `simple feature vector`
* Histogram of Oriented Gradients (HoG)--Hubel and Wiesel found these oriented edges---edges  
Eg: Bag of Words  
    Step1: Build Codebook: atches-->Cluster patches to form "codebook" of "visual words".
    Step2: Encode Images
    
###### Image features vs ConvNet
Image-->Extraction-->f-->10 numbers giving scores for classes-->trainig-->Feature Extraction
Image-->ConvNet-->10 numbers giving scores for calsses-->training-->ConvNet

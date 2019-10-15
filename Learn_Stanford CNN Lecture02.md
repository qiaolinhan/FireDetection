
## Lecture 3
##### RECALL
Loss functions, Optimization  
Challenges of recognition: Viewpoint, Illumination, Deformation, Occlision, Clutter, Intraclass Variation.  
Data-driven approach, K-Nearest Neighbors (KNN).  
Linear Classifier $ f(x,W)=Wx+b $ Outputs the scores.  
  * How to actually chosse the W?
  1. Define a loss function that quatifies our unhappiness with the score across the training data.  
  2. Come up with a way of efficiently finding the parameters that minimize the loss function. (*Optimization*)  
  
$ f(x,W)=Wx $  
A *Loss function* tells how good our current classifier is  
Given a dataset of examples:  
$ {(x_i,y_i)}_{i=1}^N $  
Where $ x_i $ is image and $ y_i $ is (integer) $ lable   
Loss over the dataset is sum of loss over examples:
$ L={\frac{1}{n} \sum_{i}L_i(f(x_i,W),y_i)} $  
* To concrete loss function, there is a multi-class SVM loss.  
It is a generalization of SVM to handle multiple classes.
Given an example $(x_i,y_i)$ where $ x_i $ is the image and where $ y_i $ is the lable (integer), and using the shorthand for the score vector: $ s=f(x_i,W) $  
The SVM loss has the form:
$$ L_i=\sum_{j\neq y_i} {0\enspace if s_{y_i}\geq s_j+1 \\ s_j-s_{y_i}+1\enspace otherwise} $$
$$ =\sum_{j\neq y_i}max（0，s_j-s_{y_i}+1) $$  
Where $ s $ is the  predict score, $ s_{y_i} $ is the score of true class.
* Hinge loss $ \uparrow $
* Only concern about the different scores 
 
* squared loss: mistake will be very very bad!
* hinge loss: we don't actually care between being a little wrong and being a lot wrong.
$ L_i=\sum_{j\neq y_i}max(0,s_j-s_{y_i}+1) $
```
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

$ L(W)=\frac{1}{N}\sum_{i=1}^{N}L_i(f(x_i,W),y_i)+\lmbda R(W) $  
* Data loss: Model predictions should match training data， use the training data to find some classifier. `We only care about the performance on test data`
* Regularization: Model should be "simple", so it works on test data.  
* Whole idea: Occam's Razor.  

$ \lambda $ =regularization strength (hyperparameter)
###### Common Use Regularization
* #$ L2 regularization: $ R(W)=\sum_{k}\sum_{l}W_{k,l}^2 $#  
* L1 regularization: $ R(W)=\sum_{k}\sum_{l}|W_{k,l}| $ (encouraging sparity in the matrix W)
* Elastic net (L1+L2) $ R(W)=\sum_k\sum_l\beta W_{k,l}^2+|W_{k,,l}| $   (ML DL)
* Max norm regularization
* Dropout
* Fancier: Batch normalization, stochaastic depth


  


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
Given a dataset of examples $ {(x_i,y_i)}_{i=1}^N $  
Where $ x_i $ is image and $ y_i $ is (integer) lable  
Loss over the dataset is sum of loss over examples:
$ L=\frac{1}{n}\sum_{i}L_i(f(x_i,W),y_i) $  
* To concrete loss function, there is a multi-class SVM loss.  
It is a generalization of SVM to handle multiple classes.
Given an example $(x_i,y_i)$ where $ x_i $ is the image and where $ y_i $ is the lable (integer), and using the shorthand for the score vector: $ s=f(x_i,W) $  
The SVM loss has the form:
$$ L_i=\sum_{j\neq y_i} { {0& if s_{y_i}\geq s_j+1\\s_j-s_{y_i}+1&otherwise} $$
$$ =\sum_{j\neq y_i}max（0，s_j-s_{y_i}+1) $$
  

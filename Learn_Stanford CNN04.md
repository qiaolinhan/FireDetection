# Lecture 4
Backpropagation and Neural Networks 

#### Recall
* score: $ s=f(x;W)=Wx $  
* SVM loss: $ L_i=\sum_{j\neq y_i}max(0,s_j-s_{y_i}+1) $  
* data loss+ regularization: $ L=\frac{1}{N}\sum_{i=1}{N}Li+\sum_kW_k^2 $  
we want $ \bigtriangledown_WL $, we want to find the parameters _W_.  

###### Gradient Decent
$ df(x)/dx=\lim_{h\rightarrow 0}\frac{f(x+h)-f(x)}{h} $  
* Numerical gradient: slow, approximate, but easy to write.  
* Analitic gradient: error-prone, but fast and exact.  
Practice: derive analytic gradient, check your implementation with numerical gradient  

#### Computation Graphs
$ f=Wx $ ---score  
$ L_i=\sum_{j\neq y_i}max(0,s_j-s_{y_i}+1) $---hinge loss  
$ R(W) $---regularization  
$ L=\sum{L_i}+R(W) $
* Advantage: backpropagation can be used (chain rule) to compute the gradient.

Eg: $ f(x,y,z)=(x+y)z
      q=x+y, f=qz,  
      \partial q/\partial x, \partial q/\partial y   
      \partial f/\partial q=z  
      \partial f/\partial z=q $  
      We want $ \partial f/\partial x,\partial f/\partial y,\partial f/\partial z $  
      * chain rule: $ \partial f/\partial y=\frac{\partial f}{\partial q}\frac{\partial q}{\partial y} $  
      
 
Local gradient---$\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} $  

            

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
      \frac{\partial q}{\partial x},\frac{\partial q}{\partial y}   
      \frac{\partial f}{\partial q}=z, \frac{\partial f}{\partial z}=q $  
      We want $ \frac{\partial f}{\partial x},\frac{\partial f}{\partial y},\frac{\partial f}{\partial z} $  
      * chain rule: $ \partial f/\partial y=\frac{\partial f}{\partial q}\frac{\partial q}{\partial y} $  
      
##### Backpropagation 
Local gradient---$ \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} $  
gradients---$ \frac{\partial L}{\partial z} $  
Use the chain rule we can compute $ \frac{\partial L}{\partial x} and \frac{\partial L}{\partial y} $  

Eg: $ f(w,x)=\frac{1}{1+e^{-(w_0x_0+w_1x_1+w_2)}} $     
    $ e^x-->e^x; ax-->a; 1/x-->-1/x^2; x+a-->1 $    
    $ \sigma(x)=\frac{1}{1+e^{-x}}---sigmoid gate $  

##### Patterns in backward flow
* *add gate*: gradient distributor  
Q: What is the max gate?  

            

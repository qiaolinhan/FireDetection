# Tutorial for Pytorch
PyTorch is a Python-based library which facilitates building Deep Learning models and using them in various applications. But, it’s more than just another Deep Learning library, it’s a scientific computing package (as the official PyTorch docs state).  
* linked with google collab.
* Or download the notebook and run locally.  
The tutorial is broken out into 4 modules:
1. What is pytorch-tensors and multiple syntaxes
2. Autimatic differenciation
3. training the neural network
4. get to use in real life example-training classifier 

* _CLOULD_ Linode.com/sentdex which has the best cloud GPU price

* maybe work on the jupyter lab

deep learning works like millions of small calculation. CPU performs better on mroe complicated calculation; GPU - graphs - very small calculations.` _This is the reason why we run on GPU_`

## Introduction
```python
print ('hello world!')
import torch
import matplotlib.pyplot as plt
x = torch.Tensor([5,3])
y = torch.Tensor([2,1,0,
                  1,2,3])

print(x)
print(y)


x.shape

'''y = torch.rand([255,255]) # pixels as in an image
plt.plot(y)
plt.show()'''
```
## Data input to neural network


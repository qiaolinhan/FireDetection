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
* in jupyter lab  
`torch` comes with a bunch of data sets, `torchvison` is a collection of data that is used for vision  
Most of ourtime is going to be -- **getting data, preparing data, formatting data** in such a way that's gonna work with a neural network.  
Another thing we need to do is **banching**.
* It is important to seperate training data sets and a test data set as soon as possible.  
We need to convert data to tensor, so there is a `transforms.ToTensor()`
```python
import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# define our two major data sets: training data set and testing data set
# testing data should be an out-of-sample testing data
train = datasets.MNIST("", train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor()]))
# in [] we paste all the transforms we want to apply to data
test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
# batch=how many we want to pass to model at a time, usually between 8 and 64, bigger=faster.
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

for data in trainset:
    print(data)
    break
# x, y = data[0][0], data[1][0]
# print(y)
print(data[0][0].shape)
# we find that the image is not a typical, because it's 1*28*28
plt.imshow(data[0][0].view(28,28))
plt.show()
```

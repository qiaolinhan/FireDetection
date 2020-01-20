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
Let the model to learn to decrease the loss quickly, the loss is measured from the output of the NN.
* A way to confirm a dataset balance is to make a `couner`. 
```python
total = 0
counter_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

for data in trainset:
    Xs, ys = data
    for y in ys:
        counter_dict[int(y)] += 1
        total += 1
print(counter_dict)

for i in counter_dict:
    print(f"{i}:{counter_dict[i] / total * 100}")  # percentage of the distribution
```
* data is more important than the NN

## Building Neural Network
* do not forget the line `super().__init__()`
```python
class Net(nn.Module):
    def __init__(self):
        # first we need to initialize an end module
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        # fully connected layer 1, make a three layers of 64 neuron for the hidden layers
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)  # the output neutrons, we have 0-9 ten outputs
```
Then we need to difine the way data passes through the layers, a simple way is the feed-forward NN, so we define the `forward`. 
```python
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
```
Then we need the activation function, so the code above changes into:
```python
    def forward(self, x):
        x = F.relu(self.fc1(x))  # relu: rectified linear activation function
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # use the softmax for multi-class
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
```
* It is possible to set different activation functions for per neuron. Not much we can see.  
```python
net = Net()
print(net)

X = torch.rand((28, 28))
# print(X)
X = X.view(-1, 28 * 28)  # -1 specifies that the input will be an unknown shape
# output are the real predictions
output = net(X)
print(output)
```

## Deep learning with pytorch
pass the lable data and actually train the model to hopefully be able to recognize whatever it is we are passing.  
* **there are two concepts: loss and optimizer**

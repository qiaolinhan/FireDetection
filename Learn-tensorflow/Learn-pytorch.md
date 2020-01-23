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

## 1. Introduction
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
## 2. Data input to neural network
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

## 3. Building Neural Network
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

## 4. Deep learning with pytorch
pass the lable data and actually train the model to hopefully be able to recognize whatever it is we are passing.
* **there are two concepts: loss and optimizer**
* loss: how wrong is the modle, some degree of error is there
```python
import torch.optim as optim
from data_learn import *
from buildingNN_learn import *

optimizer = optim.Adam(net.parameters(), lr=0.001)  # learning rate=0.001, * decaying learning rate
# Actually we do not optimize for accuracy, we optimize for loss, it just happens that accuracy follows.
EPOCH = 3
for epoch in range(EPOCH):
    for data in trainset:
        # data is a batch of features and labels
        X, y = data
        # print(X[0])
        # print(y[0])
        # break
        net.zero_grad()  # zero the gradients so that only one batch pass because of the very weak GPU or CPU
        output = net(X.view(-1, 28*28))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)
```
We want to iterate pass our data though the model, **a full pass through our data is what is called an epoch**  
There two major ways to calculate loss:
* one-hot vectors, we use squared mean error
* if our data set is a scalar value, not a vector, just use `nll_loss`.

## 5. Convnet introduction
In the example, search "cat vs dog" in Google to download.  
2D images --> pixels --> apply these convolutions kernels --> sliding the kernel window over the entire image --> pooling window max or everage
* the goal of convolution is to locate the images
* NN works on numbers , not strings, not slants  
**The 1st layer of coonvolution kernels is to find the edges, curves or corner, then the next layer finds more complex features that edges, curves or corners build (combinations-->circles, sqaures), then next layer finds the combination of circles and square**
```python
import numpy as np
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# REBUILD_DATA = True
REBUILD_DATA = False


class DogsVSCats():
    IMG_SIZE = 50
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABLES = {CATS: 0, DOGS: 1}

    training_data = []
    catcount = 0
    dogcount = 0

    def make_training_data(self):
        for label in self.LABLES:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    self.training_data.append([np.array(img), np.eye(2)[self.LABLES[label]]])

                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1

                except Exception as e:
                    pass
                    print(str(e))

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("Cats:", self.catcount)
        print("Dogs:", self.dogcount)


if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()

training_data = np.load("training_data.npy", allow_pickle=True)
print(len(training_data))

plt.imshow(training_data[0][0], cmap="gray")
plt.show()

print(training_data[0][0])
```

## 6. Convolution layer and batching data
```python
import numpy as np
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# REBUILD_DATA = True
REBUILD_DATA = False


class DogsVSCats():
    IMG_SIZE = 50
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABLES = {CATS: 0, DOGS: 1}

    training_data = []
    catcount = 0
    dogcount = 0

    def make_training_data(self):
        for label in self.LABLES:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    self.training_data.append([np.array(img), np.eye(2)[self.LABLES[label]]])

                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1

                except Exception as e:
                    pass
                    print(str(e))

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("Cats:", self.catcount)
        print("Dogs:", self.dogcount)


if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()

training_data = np.load("training_data.npy", allow_pickle=True)
print("size of whole training data:", len(training_data))

plt.imshow(training_data[0][0], cmap="gray")
# plt.show()

# print(training_data[0][0])


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
# no idea what is the input of the full connected layer from convolution layer
# in Tensorflow there is keras and flatten
# self.fc1 = nn.Linear(-1, 512), we do not know the size of input
# there is a way as stated below
        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        print("size of input to full connected layer:", x[0].shape)
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # add the activation layer
        return F.softmax(x, dim=1)


net = Net()

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

# for here, we have our training data set and training parameters

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

# separate out some training, testing and validation date
VAL_PCT = 0.1  # validation percentage
val_size = int(len(X)*VAL_PCT)
print("validation_size:", val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

print('size of training data:', len(train_X))
print('size of test data:', len(test_X))

# Actually train
BATCH_SIZE = 100
EPOCHS = 5

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        # print(i, i+BATCH_SIZE)
        batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
        batch_y = train_y[i:i+BATCH_SIZE]

        net.zero_grad()
        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()

print(loss)

correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1, 1, 50, 50))[0]
        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class:
            correct += 1
        total += 1
print("Accuracy:", round(correct/total, 3))
```


```python
!pip3 install torch

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

X = torch.randn(100, 1)*10

y = X + 3*torch.randn(100, 1)

'''
plt.plot(X.numpy(), y.numpy(), '.')
plt.grid()
plt.ylabel('y')
plt.xlabel('x')
plt.show()
'''


class LR(nn.Module):  # linear regression
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        pred = self.linear(x)
        return pred


torch.manual_seed(1)
model = LR(1, 1)
print(model)

[w, b] = model.parameters()
print(w, b)


def get_params():
    return w[0][0].item(), b[0].item()


'''
def plot_fit(title):
    plt.title = title
    w1, b1 = get_params()
    x1 = np.array([-30, 30])
    y1 = w1*x1+b1
    plt.plot(x1, y1, 'r')
    plt.scatter(X, y)
    plt.grid()
    plt.show()


plot_fit('initial Model')
'''
criterion = nn.MSELoss()  # remember the brank
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 100
losses = []
for i in range(epochs):
    y_pred = model.forward(X)
    # print(y_pred)
    # print(y)

    loss = criterion(y_pred, y)
    print('epoch:', i, 'loss:', loss.item())

    # to visualize the decrease of the loss at every single epoch
    losses.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid()
plt.show()
```
<img scr="https://github.com/qiaolinhan/FireDetection/blob/master/Learn-tensorflow/imgs/LR.png">

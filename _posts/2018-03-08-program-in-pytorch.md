---
layout: post
title: "Program in PyTorch"
categories: []
tags: [machine-learning, pytorch, library]
description: PyTorch is an open source machine learning library for Python, based upon Torch, an open-source machine learning library, a scientific computing framework, and a script language based on Lua programming language.
cover: "/assets/images/pytorch.jpg"
cover_source: "https://mesosphere.com/wp-content/uploads/2017/12/pytorch-header.jpg"
comments: true
mathjax: true
---

### Introduction

Pytorch provides two high-level features:

* Tensor computation analogous to numpy but with option of GPU acceleration.
* Deep Neural Networks built on a tape-based autograd system.

And is generally used either as replacement for NumPy (for the GPU acceleration) or as a deep learning research platform.

### Components of PyTorch

* **torch**: Tensor library like NumPy with GPU support.
* **torch.autograd**: Automatic differentiation library that supports all differentiable Tensor operations.
* **torch.nn**: neural network library integrated with autograd.
* **torch.optim**: optimization package used along with torch.nn with standard optimization methods like SGD, RMSProp, LBGFS etc.
* **torch.multiprocessing**: python multiprocessing, but with memory sharing of Tensors across processes.
* **torch.utils**: Data loader, trainer and other utility functions.
* **torch.legacy(.nn/.optim)**: legacy code that is ported for backward compatibility.

### Tensors

Tensors in torch are analogous to ndarrays in NumPy but differ in that Tensors in torch can be loaded on to GPU for hardware acceleration.

Tensors can be initialized by calling a normal Tensor object or using special purpose functions like `torch.rand`. The `size` function gives the dimension of the Tensor initialized.

> Unlike Tensors in TensorFlow, the ones in PyTorch can be seen after initialization without running a session.

```python
# for python 2.* users
from __future__ import print_function
import torch

x = torch.Tensor(4, 3)
print(x)
x = torch.rand(4, 3)
print(x)

print(x.size())
```

### Tensor Operations

PyTorch gives various options and aliases for operations on tensors as can be seen for addition below

```python
y = torch.rand(4, 3)

# addition using + operator
print(x + y)

# addition using add function
print(torch.add(x, y))

# addition using out parameter of add function
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
# result is casted for the new dimensions [4*3]
print(result)

# in-place addition 
y.add_(x)
print(y)
```

Standard NumPy-like indexing works on PyTorch Tensors.

```python
print(x[:, 1])
```

Resizing can be done using `torch.view`.

```python
x = torch.randn(4, 4)
y = x.view(16)
# the size -1 is inferred from other dimensions
z = x.view(-1, 8)  
print(x.size(), y.size(), z.size())
```

Some operations available on Tensors are:

* **torch.numel**: returns the total number of elements in a Tensor.
* **torch.eye**: returns a 2D tensor representing an identity matrix.
* **torch.from_numpy**: create a Tensor from NumPy array where the two share the same memory and modifications are reflected across.
* **torch.linspace**: returns a 1D tensor of equally spaced steps with start and end of a range.
* **torch.ones**: returns tensor of a defined shape filled with scalar value 1.
* **torch.zeros**: returns tensor of a defined shape filled with scalar value 0.
* **torch.cat**: concatenate tensors.
* **torch.chunk**: splits the tensors into chunks.

### CUDA Tensors

* Moving Tensors to CUDA is as simple as calling `.cuda` method.
* Calling a simple function, `torch.cuda.is_available` checks if CUDA is available.

> The type of a variable moved to GPU differs from the ones not on GPU, and hence addition would lead to TypeError.

```python
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    print(x+y)
x = torch.rand(4, 3)
# raises TypeError because of Type Mismatch [torch.FloatTensor, torch.cuda.FloatTensor]
print(x + y)
```

### Autograd: Automatic Differentiation

As the name suggests, the `autograd` package provides automatic differentiation for all operations on Tensors. The **define-by-run framework** ensures that the backprop is defined by how the code is run, and hence every single iteration can be different allowing dynamic modifications between epochs during training which is not possible in other static libraries like TensorFlow, Theano etc. which require a graph compilation by running something like a session and any change in network requires a recompilation of the graph.

### Variable

`autograd.Variable` is the main class under the autograd package, which wraps a tensor along with almost all of operations defined on it. Upon completion of a process, `.backward` method can be called to calculate all the gradients in the backward direction making the back propagation a very minor automatic step in designing the network.

* `.data` gives the raw tensor in the variable.
* `.grad` gives the gradient w.r.t. this variable.

The other important class in autograd package is the `Function` class. `Variable` and `Function` are interconnected to build an **acyclic graph**, the encodes the complete history of computation.

> Every variable has a `.grad_fn` attribute that references the `Function` that has created the `Variable`. The variables created by user have `grad_fn` as `None`.

In order to compute derivatives, `.backward` function can be called on a variable.

* if `Variable` is a scalar, `.backward` does not require any argument.
* if `Variable` holds more types of elements, a `gradient` argument is defined, which is a tensor of a matching shape.

```python
import torch
from torch.autograd import Variable

x = Variable(torch.ones(2, 2), requires_grad=True)
y = x + 2
print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)

out.backwards()
print(x.grad)
```

Let the `out` variable be \\(o\\), then,

$$o = {1 \over 4} \sum_i z_i \tag{1} \label{1}$$

where \\(z_i\\) is given by,

$$z_i = 3(x_i + 2)^2 \tag{2} \label{2}$$

So \\(o\\) in \eqref{1} can be written as,

$$o = {1 \over 4} \cdot 3(x_i + 2)^2 \tag{3} \label{3}$$

Differentiating w.r.t. x,

$${\partial o \over \partial x_i} = {1 \over 4} \cdot 3 \cdot 2 (x_i + 2) = {3 \over 2} (x_i + 2) \tag{4} \label{4}$$

The autograd package can be in general seen as the library that implements these basic differentials and then carefully employs the chain rule to generate gradients for the most complex functions in the program also. This simplifying of the process helps achieve gradients on the fly instead of predefining it for a graph.

### Neural Networks

The neural network package, `torch.nn`, depends heavily on the `autograd` package to define the models and differentiate them. 

A basic training process in the neural network involves the following steps:

* define the neural network with learnable parameters (i.e. weights)
* iterate over the training data.
* feed the input through the network.
* compute the loss or other error metrics.
* propagate the gradients back into the network parameters.
* update the weights.

Define a network:

```python
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# inherit from nn.Module
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
```

Only the forward function needs to be defined because the backward function is already defined using autograd. And the learnable parameters of the network are returned by `net.parameters`.

```python
params = list(net.parameters())
print(len(params))
print(params[0].size()) 
```

The input to the forward method is `autograd.Variable`, and so is the output.

```python
input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
print(out)
```

The gradients of all the parameters should be reset to zero before calling the backprops.

```python
net.zero_grad()
out.backward(torch.randn(1, 10))
```

> `torch.nn` only supports mini-batches, i.e. input to any nn layer is a 4D Tensor of `samples * channels * height * width`. If a single sample, `input.unsqueeze(0)` adds a fake dimension.

### Loss Function

> A loss function takes (output, target) pair of inputs, and computes a value that estimates how far away the output is from the target.

```python
output = net(input)
target = Variable(torch.arange(1, 11))  # a dummy target, for example
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
```

Using the `.grad` and `.next_functions` one can see the entire graph in the backward direction from the loss. Upon calling the `.backward` function the whole graph is differentiated w.r.t. the loss, and all `.grad` variables are updated with the accumulated gradients.

Before calling the backward function, existing gradients need to be cleared or they will be accumulated along with the existing gradients, if any.

```python
net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
```

The simplest way to update the weights is the Stochastic Gradient Descent (SGD).

$$ weight = weight - learning_rate * gradient \tag{5} \label{5}$$

```python
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
```

Apart from this, PyTorch also allows to use various other algorithms for the purpose of optimizing the network parameters. This is enabled by using the `torch.optim` package that implements most of these methods.

```python
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
```

It can be seen that, even the optimizer package requires the manual reseting of the gradient buffer before backpropagation is invoked, to prevent the accumulation of gradients from different calls.

### Handling Data

Data may deal with one of the formats, namely, image, text, audio or video. Standard python packages may be used to load the datasets into the NumPy arrays, which can then easily be converted into the Tensors because of the seamless bridge between the two libraries.

Following packages are recommended:

* **Images**: Pillow, OpenCV
* **Audio**: scipy, librosa
* **Text**: raw Python or Cython based loading, or NLTK and SpaCy

> Specifically for vision, the package torchvision is defined, that has loaders for common datasets such as Imagenet, CIFAR10, MNIST etc. and data transformers for images.


## REFERENCES:

<small>[PyTorch Official Page](http://pytorch.org/){:target="_blank"}</small><br>
<small>[PyTorch - Wikipedia](https://en.wikipedia.org/wiki/PyTorch){:target="_blank"}</small><br>
<small>[PyTorch - Tensor](http://pytorch.org/docs/master/torch.html){:target="_blank"}</small><br>
<small>[Torch (machine learning) - Wikipedia](https://en.wikipedia.org/wiki/Torch_(machine_learning)){:target="_blank"}</small>

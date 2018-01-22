---
layout: post
title: "Understanding TensorFlow"
categories: []
tags: [machine learning, tensorflow]
description: TensorFlow is an open source, data flow graph based, numerical computation library. Nodes in the graph represent mathematical operations, while edges represent the multidimensional data arrays communicated between them.
cover: "/assets/images/convolution.jpg"
cover_source: "https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580f8f75_8-convolutional-neural-networks/8-convolutional-neural-networks.jpg"
comments: true
mathjax: true
---

### Prerequisites

* Python Programming
* Basics of Arrays
* Basics of Machine Learning

### TensforFlow APIs

* The lowest level TensorFlow API, **TensorFlow Core**, provides the complete programming control, recommended for machine learning researchers who require fine levels of control over their model.
* The higher level TensorFlow APIs are built on top of TensorFlow Core. These APIs are easier to learn and use. Higher level APIs also provide convinient wrappers for repetitive tasks, e.g. **tf.estimator** helps to manage datasets, estimators, training, inference etc.

### Terminologies

* **Tensor:** A set of primitive types shaped into an array of any number of dimensions. It can be considered a higher-dimensional vector.
* **Rank:** The number of dimensions of a tensor (similar to rank of a matrix).
* **Computational Graph** defines the series of TensorFlow operations arranged in the form of graph nodes.
* **Node** in a TensorFlow graph takes zero or more tensors as inputs and produces a tensor as an output, e.g. **Constant** is a type of node in TensorFlow that takes no inputs and outputs the value that it stores internally (as defined in the defination of Tensorflow nodes earlier).
* **Session**, used for evaluation of TensorFlow graphs, encapsulates the control and state of the TensorFlow runtime.
* **Operations** are another kind of node in TensorFlow used to build the computational graphs.
* **TensorBoard** is a TensorFlow utility used to visualize the computational graphs.
* **Placeholder** is a promise to provide value later. These serve the purpose or parameters or arguments to a graph which represents a dynamic function based on inputs.

### TensorFlow Program

Basically, a TensorFlow Core program can be split into two sections:

* Building the computational graph
* Running the computational graph

Tensor initialization is done as follows:

```python
import tensorflow as tf
node1 = tf.constant(3.0, dtype=tf.float32) #1
node2 = tf.constant(4.0) #2
print(node1, node2)
```

The print statement does not print value assigned to the nodes. The actual values will be displayed only on evaluation of the nodes. In TensorFlow the evaluation of a node can only be done within a session as shown below.

```python
sess = tf.Session()
print(sess.run([node1, node2]))
```
More complex TensorFlow graphs are built using operation nodes. For example,

```python
node3 = tf.add(node1, node2)
print(node3)
print(sess.run(node3))
```
In order to visualize the TensorFlow graph, do as follows:

```python
tf.summary.FileWriter('/path/to/save/', sess.graph)
```

Now run the following command on the terminal, ensure **TensorBoard** is installed.

```shell
tensorboard --logdir=/path/to/save
```

![TensorFlow Graph Visualization](/assets/2018-01-02-understanding-tensorflow/fig-1-tensorflow-graph-visualization.png?raw=true)


It leads to the following graph display in the tensorboard window, which is an apt representation of the minimal graph that has been built so far. But it is a constant graph as the input nodes are constants. In order to build a parameterized graph, placeholders are used as shown below.

```python
import tensorflow as tf

a = tf.placeholder(tf.float32, name='a')
b = tf.placeholder(tf.float32, name='b')
adder = a+b

sess = tf.Session()
sess.run(adder, {a: 3, b: 4})

tf.summary.FileWriter('./', sess.graph)
```
![TensorFlow Placeholder Graph](/assets/2018-01-02-understanding-tensorflow/fig-2-placeholder-graph.png?raw=true)

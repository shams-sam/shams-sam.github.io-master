---
layout: post
title: "How to train your Neural Network?"
categories: [basics-of-machine-learning]
tags: [machine-learning, andrew-ng]
description: Basic set of steps to follow while training a neural network. A lot of these are just recommendations. There are no rules in the world of neural networks.
cover: "/assets/images/network-architecture.jpg"
cover_source: "https://outlierventures.io/wp-content/uploads/2018/03/deeplearningmeetup.jpg"
comments: true
mathjax: true
---

{% include collection-basics-of-machine-learning.md %}

### Pick a network architecture:

This usually means to pick the connectivity pattern between the neurons.

![Fig.1 - Network Architectures](/assets/2018-03-31-how-to-train-your-neural-network/fig-1-network-architectures.png?raw=true)

Fig.1 shows a few examples of network architectures. It can be seen that the three networks have the same number of input and output units. This is so because the input units equals input features in a given training example while the output units equals the number of target classes. 

> As a general rule, the when the number of classes are greater than 2, then the classes should be one hot encoded to ease the process of classification. It can be seen an defining individual logistic units for determining whether or not it belongs to that class.

As a default, the number of hidden layers can be kept as 1, or if the number of hidden layers is greater than 1, then the same number of hidden units in each layer. Also, the more the number of hidden units the better. One drawback it presents is in terms of computation expense. The number of units in a hidden layer is generally comparable or a little greater than the number of features (2, 3, or 4 times maybe).

### Training a neural network:

* [Random weight initialization]({% post_url 2018-03-29-backpropagation-implementation-and-gradient-checking %}#random-initialization)
* Forward propagation to calculate \\(h(x^{(i)})\\)
* Calculate the cost, \\(J(\Theta)\\)
* [Backpropagation]({% post_url 2017-10-03-neural-networks-cost-function-and-back-propagation %}#backpropagation-algorithm) to compute partial derivatives, \\(\frac {\partial} {\partial \Theta_{jk}^{(l)}} J(\Theta)\\)
* Use [gradient checking]({% post_url 2018-03-29-backpropagation-implementation-and-gradient-checking %}#gradient-checking) to compare \\(\frac {\partial} {\partial \Theta_{jk}^{(l)}} J(\Theta)\\) given by backpropagation vs the numerical estimate of gradient. Then disable gradient checking.
* Use gradient descent or any other optimization algorithm to minimize \\(J(\Theta)\\).

> For neural networks the cost function, \\(J(\Theta)\\) is non-convex and hence susceptible to local minima. But in practice it does not present a serious problem in the implementation.

## REFERENCES:

<small>[Machine Learning: Coursera - Gradient Checking](https://www.coursera.org/learn/machine-learning/lecture/Wh6s3/putting-it-together){:target="_blank"}</small>

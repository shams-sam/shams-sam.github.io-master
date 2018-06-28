---
layout: post
title: "Large Scale Learning"
categories: [basics-of-machine-learning]
tags: [machine-learning, andrew-ng]
description: With the increase in size of training data, it becomes important to optimize algorithm and parallelize processes to minimize the training time and manage resource utilization.
cover: "/assets/images/large-dataset.jpg"
cover_source: "http://manovich.net/index.php/projects/cultural-analytics-of-large-datasets-from-flickr"
comments: true
mathjax: true
---

### Introduction

The popularity of machine learning techniques have increased in the recent past. One of the reasons leading to this trend is the exponential growth in data available to learn from. Large datasets coupled with a high variance model has the potential to perform well. But as the size of datasets increase, it poses various problems in terms of space and time complexities of the algorithms. 

> It's not who has the best algorithm that wins. It's who has the most data.

For example, consider the update rule for parameter optimization using gradient descent from (3) and (4) in the [multivariate linear regression post]({% post_url 2017-08-23-multivariate-linear-regression %}){:target="\_blank"}, 

$$\theta_j := \theta_j - \alpha {1 \over m} \sum_{i=1}^m \left( h_{\theta}(x^{(i)}) - y^{(i)} \right) x_j^{(i)} \tag{1} \label{1}$$

> [Kaggle Kernel Implementation](https://www.kaggle.com/shamssam/gradient-descent-for-regression){:target="\_blank"}

```python
def batch_update_vectorized(self):
    m, _ = self.X_train.size()
    return torch.matmul(
            self._add_bias(self.X_train).transpose(0, 1),
            (self.forward() - self.y_train)
        ) / m

def batch_update_iterative(self):
    m, _ = self.X_train.size()
    update_theta = None
    X = self._add_bias(self.X_train)
    for i in range(m):
        if type(update_theta) == torch.DoubleTensor:
            update_theta += (self._forward(self.X_train[i].view(1, -1)) - self.y_train[i]) * X[i]
        else:
            update_theta = (self._forward(self.X_train[i].view(1, -1)) - self.y_train[i]) * X[i]
    return update_theta/m
    

def batch_train(self, tolerance=0.01, alpha=0.01):
    converged = False
    prev_cost = self.cost()
    init_cost = prev_cost
    num_epochs = 0
    while not converged:
        self.Theta = self.Theta - alpha * self.batch_update_vectorized()
        cost = self.cost()
        if (prev_cost - cost) < tolerance:
            converged = True
        prev_cost = cost
        num_epochs += 1
```

From \eqref{1} above, it can be seen that for each step of gradient descent, summation has to be performed over entire dataset of \\(m\\) examples. While for small datasets it might seem inconsequential, but as the size of datasets increases this would have very high impact on the training time. 

In such cases, it would also be helpful to plot [learning curves]({% post_url 2018-04-02-evaluation-of-learning-algorithm %}#learning-curves), to check if actually training the model with such high number data samples is really helpful, because if the model has high bias then similar result could be acheived by using a smaller dataset. It would be more helpful to incrase variance of the model in such cases.

On the other hand, if the learning curves show that using the larger dataset is indeed helpful, it would be more productive to use more computationally efficient algorithms to train the model such as the ones mentioned in the following sections.

### Stochastic Gradient Descent

The gradient descent rule presented in \eqref{1}, also known as **batch gradient descent**, has the disadvantage that for each update the summation of update term has to be performed over all the training data. 

Stochastic gradient descent is an approximation of the batch gradient descent. Each epoch in this algorithm is begun with a random shuffle of the data followed by the following update rule,

$$\theta_j := \theta_j - \alpha \left( h_{\theta}(x^{(i)}) - y^{(i)} \right) x_j^{(i)} \tag{2} \label{2}$$

```python
def stochastic_train(self, tolerance=0.01, alpha=0.01):
    converged = False
    m, _ = self.X_train.size()
    X = self._add_bias(self.X_train)
    init_cost = self.cost()
    num_epochs=0
    while not converged:
        prev_cost = self.cost()
        for i in range(m):
            self.Theta = self.Theta - alpha * (self._forward(self.X_train[i].view(1, -1)) - self.y_train[i]) * X[i]
        cost = self.cost()
        if prev_cost-cost < tolerance:
            converged=True
        num_epochs += 1
```

i.e. for each training data in the sample dataset, as soon as the cost correponding to that instance is calculated it is used to make an approximate update to the parameters instead of waiting for the summation to finish. While this is not as accurate as the batch gradient descent in reaching the global minimum, it always converges within its close proximity.

> In practice, stochastic gradient descent speeds up the process of convergence over the traditional batch gradient descent.

While learning rate is kept constant in most implementations of stochastic gradient descent, it is observed in practice that it helps to taper off the value of learning rate as the iteration proceeds. It can be done as follows,

$$\alpha = \frac {constant_1} {iteration\_number + constant_2} \tag{3} \label{3} $$

### Mini-Batch Gradient Descent

While batch gradient descent sums over all the data for a single update iteration of the parameters, the stochastic gradient descent does it by considering individual training examples as and when they are encountered. The **mini-batch gradient descent** takes the mid-way and uses the summation from only **b training examples (i.e. batch size)** for every update iteration. Mathematically it can be presented as follows,

$$\theta_j := \theta_j - \alpha {1 \over b} \sum_{i=1}^{i+b} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right) x_j^{(i)} \tag{4} \label{4}$$

```python
def mini_batch_train(self, tolerance=0.01, alpha=0.01, batch_size=8):
    converged = False
    m, _ = self.X_train.size()
    X = self._add_bias(self.X_train)
    init_cost = self.cost()
    num_epochs=0
    while not converged:
        prev_cost = self.cost()
        for i in range(0, m, batch_size):
            self.Theta = self.Theta - alpha / batch_size * torch.matmul(
                X[i:i+batch_size].transpose(0, 1),
                self._forward(self.X_train[i: i+batch_size]) - self.y_train[i: i+batch_size]
            )
        cost = self.cost()
        if prev_cost-cost < tolerance:
            converged=True
        num_epochs += 1
```

* Compared to stochastic gradient descent, the mini-batch gradient descent will be faster only if vectorized implementation is used for the updates.

* Compared to batch gradient descent, the mini-batch gradient descent is faster due to the obvious reason of lesser number of summations that are to be performed for a single update iteration. Also, if both the implementations are vectorized, mini-batch gradient descent will have lower memory usage. The speed of operations depends on the trade-off between the matrix operation complexities and memory usage. 

* Generally it is observed that mini-batch gradient descent converges faster than both stochastic and batch gradient descent.

### Online Learning

Online learning is a form of learning when the system has a continuous stream of training data. It implements the stochastic gradient descent forever using the input stream of data and discarding it once the parameter updates have been done using it.

It is observed that such an online learning setting is **capable of learning the changing trends** of data streams.

Typical domains where online learning can be successfully implemented include, search engines (predict click through rate i.e. CTR), recommendation websites etc.

Many of the listed problems can be modeled as a standard learning problem with fixed dataset, but often such data streams are available in such abundance that there is little utility of storing the data in place of implementing an online training system.

### Map Reduce and Parallelism

Map-Reduce is a technique used in large scale learning when a single system is not enough to train the models required. Under this training paradigm, all the **summation operations are parallelized over a set of slave systems by spliting the training data** (batch or entire set) across the systems which compute on smaller datasets and feed the results to the **master system that aggregates the results** from all the slaves and combines them together. This parallelized implementation boosts the speed of algorithm.

If the network latencies are not high, then one can expect a boost in speed by upto \\(n\\) times by using a pool of \\(n\\) systems. So, in practice when the systems are on a network speed boost is slightly less than \\(n\\) times.

> Algorithms that can be expressed as a summation over the training sets can be parallelized using map-reduce.

Besides a pool of computers, parallelization also works on multi-core machines with the added benifit of near-zero network latencies and hence faster. 

## REFERENCES:

<small>[Machine Learning: Coursera - Learning with Large Dataset](https://www.coursera.org/learn/machine-learning/lecture/CipHf/learning-with-large-datasets){:target="_blank"}</small><br>
<small>[Machine Learning: Coursera - Stochastic Gradient Descent](https://www.coursera.org/learn/machine-learning/lecture/DoRHJ/stochastic-gradient-descent){:target="_blank"}</small><br>
<small>[Machine Learning: Coursera - Mini-Batch Gradient Descent](https://www.coursera.org/learn/machine-learning/lecture/9zJUs/mini-batch-gradient-descent){:target="_blank"}</small><br>
<small>[Machine Learning: Coursera - Convergence of Stochastic Gradient Descent](https://www.coursera.org/learn/machine-learning/lecture/fKi0M/stochastic-gradient-descent-convergence){:target="_blank"}</small><br>
<small>[Machine Learning: Coursera - Online Learning](https://www.coursera.org/learn/machine-learning/lecture/ABO2q/online-learning){:target="_blank"}</small><br>
<small>[Machine Learning: Coursera - Map Reduce and Data Parallelism](https://www.coursera.org/learn/machine-learning/lecture/10sqI/map-reduce-and-data-parallelism){:target="_blank"}</small>

---
layout: post
title: "Recommender Systems"
categories: [basics-of-machine-learning]
tags: [machine-learning, andrew-ng]
description: A recommender system or a recommendation system is a subclass of information filtering system that seeks to predict the "rating" or "preference" a user would give to an item.
cover: "/assets/images/recommendation.jpeg"
cover_source: "https://cdn-images-1.medium.com/max/1920/1*YsJVKieG3EEA8BrWMI9ljw.jpeg"
comments: true
mathjax: true
---

{% include collection-basics-of-machine-learning.md %}

### Problem Formulation

Give \\(n_m\\) choices and \\(n_u\\) users, 

* \\(r(i, j) = 1\\) if user \\(j\\) has rated choice \\(i\\).
* \\(y(i,j)\\) is the rating given by user \\(j\\) to the choice \\(i\\), defined only if \\(r(i, j) = 1\\).

So, the objective of the reocmmender system is to use the rated choices by the population of users and predict the ratings that a user would attribute to a choice that is not rated i.e. \\(r(i, j) = 0\\). In most real-world cases such as movie ratings, the number of unrated choices is generally very high and hence is not an elementary/easy problem to solve.

### Content Based Recommendations

* Each choice is alloted an \\(n\\) number of features and rated along those dimensions.
* Following this, for each user \\(j\\) the ratings are regressed as a function of the alloted set of features.
* The learnt parameter for user \\(j\\), \\(\theta^{(j)}\\) lies in space \\(\mathbb{R}^{n+1}\\).

Summarizing, 

* \\(\theta^{(j)}\\) is the parameter vector for user \\(j\\).
* \\(x^{(i)}\\) is the feature vector for choice \\(i\\).
* For user \\(j\\) and choice \\(i\\), predicted rating is given by, \\((\theta^{(j)})^T (x^{(i)})\\).

Suppose user \\(j\\) has rated \\(m^{(j)}\\) choices, then learning \\(\theta^{(j)}\\) can be treated as linear regression problem. So, to learn \\(\theta^{(j)}\\), 

$$min_{\theta^{(j)}} {1 \over 2} \sum_{i: r(i, j)=1} ((\theta^{(j)})^T (x^{(i)}) - y^{(i, j)})^2 + {\lambda \over 2} \sum_{k=1}^n (\theta_k^{(j)})^2 \tag{1} \label{1}$$

Similarly, to learn \\(\theta^{(1)}, \theta^{(2)}, \cdots, \theta^{(n_u)}\\), 

$$min_{\theta^{(1)}, \cdots, \theta^{(n_u)}} {1 \over 2} \sum_{j=1}^{n_u} \sum_{i: r(i, j)=1} ((\theta^{(j)})^T (x^{(i)}) - y^{(i, j)})^2 + {\lambda \over 2} \sum_{j=1}^{n_u} \sum_{k=1}^n (\theta_k^{(j)})^2 \tag{2} \label{2}$$

where cost function is given by,

$$J(\theta^{(1)}, \cdots, \theta^{(n_u)}) = {1 \over 2} \sum_{j=1}^{n_u} \sum_{i: r(i, j)=1} ((\theta^{(j)})^T (x^{(i)}) - y^{(i, j)})^2 + {\lambda \over 2} \sum_{j=1}^{n_u} \sum_{k=1}^n (\theta_k^{(j)})^2 \tag{3} \label{3}$$

Gradient Descent Update,

$$
\begin{align}
\theta_k^{(j)} &= \theta_k^{(j)} - \alpha \left( \sum_{i: r(i, j)=1} ((\theta^{(j)})^T x^{(i)} - y^{(i, j)}) x_k^{(i)}  \right) \text{, for } k = 0 \\
\theta_k^{(j)} &= \theta_k^{(j)} - \alpha \left( \sum_{i: r(i, j)=1} ((\theta^{(j)})^T x^{(i)} - y^{(i, j)}) x_k^{(i)} + \lambda \theta_k^{(j)}\right) \text{, otherwise }
\end{align}
\tag{4} \label{4}
$$

where, 

$$
\begin{align}
\frac {\partial} {\partial \theta_k^{(j)}} J(\theta^{(1)}, \cdots, \theta^{(n_u)}) &= \sum_{i: r(i, j)=1} ((\theta^{(j)})^T x^{(i)} - y^{(i, j)}) x_k^{(i)} \text{, for } k = 0 \\
\frac {\partial} {\partial \theta_k^{(j)}} J(\theta^{(1)}, \cdots, \theta^{(n_u)}) &= \sum_{i: r(i, j)=1} ((\theta^{(j)})^T x^{(i)} - y^{(i, j)}) x_k^{(i)} + \lambda \theta_k^{(j)} \text{, otherwise }
\end{align}
\tag{5} \label{5}
$$

Note: By convention, the terms \\({1 \over m^{(j)}}\\) terms are removed from the equations in recommendation systems. But these do not affect the optimization values as these are only constants used for ease of derivations in linear regression cost function.

> The effectiveness of content based recommendation depends of identifying the features properly, which is often not easy.

### Collaborative Filtering

> Collaborative filtering has the intrinsic property of feature learning (i.e. it can learn by itself what features to use) which helps overcome drawbacks of content-based recommender systems.

Given the scores \\(y(i, j)\\) for a choice, \\(i \in [1, n_m]\\) by various users \\(j \in [1, n_u]\\), and the parameter vector \\(\theta^{(j)}\\) for user \\(j\\), the algorithm learns the values for the features \\(x^{(i)}\\) applying regression by posing the following optimization problem,

$$min_{x^{(i)}} {1 \over 2} \sum_{j:r(i,j)=1} \left[ (\theta^{(j)})^T x^{(i)} - y(i,j) \right]^2 + {\lambda \over 2} \sum_{k=1}^n \left( x_k^{(i)} \right)^2 \tag{6} \label{6}$$

Intuitively this boils down to the scenario where given a choice and its ratings by various users and their parameter vectors, the collaborative filitering algorithm tries to find the most optimal features to represent the choice such that the squared error between the two is minimized. Since this is very similar to the linear regression problem, regularization term is introduced to prevent overfitting of the features learnt. Similarly by extending this, it is possible to learn all the features for all the choices \\(i \in [1, n_m]\\), i.e. given \\( \theta^{(1)}, \theta^{(2)}, \cdots, \theta^{(n_u)} \\) learn, \\(x^{(1)}, x^{(2)}, \cdots, x^{(n_m)}\\),

$$min_{x^{(1)}, \cdots, x^{(n_m)}} {1 \over 2} \sum_{i=1}^{n_m} \sum_{j:r(i,j)=1} \left[ (\theta^{(j)})^T x^{(i)} - y(i,j) \right]^2 + {\lambda \over 2} \sum_{i=1}^{n_m} \sum_{k=1}^n \left( x_k^{(i)} \right)^2 \tag{7} \label{7}$$

Where the updates to the feature vectors will be given by,

$$x_k^{(i)} := x_k^{(i)} - \alpha \left( \sum_{j:r(i,j)=1} \left[ (\theta^{(j)})^T x^{(i)} - y(i,j) \right] \theta_k^{(j)} + \lambda x_k^{(i)} \right)$$

## REFERENCES:

<small>[Machine Learning: Coursera - Problem Motivation](https://www.coursera.org/learn/machine-learning/lecture/V9MNG/problem-motivation){:target="_blank"}</small><br>
<small>[Machine Learning: Coursera - Gaussian Distribution](https://www.coursera.org/learn/machine-learning/lecture/ZYAyC/gaussian-distribution){:target="_blank"}</small><br>
<small>[Machine Learning: Coursera - Algorithm](https://www.coursera.org/learn/machine-learning/lecture/C8IJp/algorithm){:target="_blank"}</small><br>
<small>[Machine Learning: Coursera - Anomaly Detection vs Supervised Learning](https://www.coursera.org/learn/machine-learning/lecture/Rkc5x/anomaly-detection-vs-supervised-learning){:target="_blank"}</small><br>
<small>[Machine Learning: Coursera - Multivariate Gaussian Distribution](https://www.coursera.org/learn/machine-learning/lecture/Cf8DF/multivariate-gaussian-distribution){:target="_blank"}</small>

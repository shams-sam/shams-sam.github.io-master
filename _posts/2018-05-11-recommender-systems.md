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

[**Kaggle Kernel**](https://www.kaggle.com/shamssam/recommender-systems){:target="\_blank"}

```python
import numpy as np

# defining a ratings matrix, Y where 0's denote not rated
y = np.array(
    [
        [3. , 0. , 4.5, 4. , 2. ],
        [3. , 4. , 3.5, 5. , 3. ],
        [0. , 0. , 3. , 5. , 3. ],
        [4. , 0. , 3. , 0. , 0. ],
        [0. , 0. , 5. , 5. , 3.5],
        [0. , 0. , 5. , 4. , 3.5],
        [0. , 5. , 5. , 5. , 4.5],
        [4. , 4. , 2.5, 5. , 0. ],
        [0.5, 0. , 4. , 0. , 2.5],
        [0. , 0. , 0. , 4. , 0. ]
    ]
)

# calculating matrix R from matrix Y
r = np.where(y > 0, 1, 0)
```

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

```python
def estimate_theta_v2(y, max_k=2, x=None, theta=None,
               _alpha = 0.01, _lambda=0.001, _tolerance = 0.001):
    r = np.where(y > 0, 1, 0)
    converged = False
    max_i, max_j = y.shape
    if type(x) != np.array:
        x = np.random.randn(max_i, max_k)
    if type(theta) != np.array:
        theta = np.random.randn(max_j, max_k+1)
    while not converged:
        update_theta = np.zeros(theta.shape)
        update_theta = _alpha * (
            np.matmul(
                np.hstack((np.ones((x.shape[0], 1)),x)).transpose(),
                (
                    np.matmul(
                        np.hstack((np.ones((x.shape[0], 1)),x)), 
                        theta.transpose()
                    ) - y
                ) * r, 
            ).transpose() + _lambda * theta
        )
        theta = theta - update_theta
        if np.max(abs(update_theta)) < _tolerance:
            converged = True
    return theta, x
```

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

$$x_k^{(i)} := x_k^{(i)} - \alpha \left( \sum_{j:r(i,j)=1} \left[ (\theta^{(j)})^T x^{(i)} - y(i,j) \right] \theta_k^{(j)} + \lambda x_k^{(i)} \right) \tag{8} \label{8}$$

```python
def estimate_x_v2(y, max_k=2, x=None, theta=None,
               _alpha = 0.01, _lambda=0.001, _tolerance = 0.001):
    r = np.where(y > 0, 1, 0)
    converged = False
    max_i, max_j = y.shape
    if type(x) != np.array:
        x = np.random.randn(max_i, max_k)
    if type(theta) != np.array:
        theta = np.random.randn(max_j, max_k+1)
    while not converged:
        update_x = np.zeros(x.shape)
        update_x = _alpha * (
            np.matmul(
                (
                    np.matmul(
                        np.hstack((np.ones((x.shape[0], 1)),x)), 
                        theta.transpose()
                    ) - y
                ) * r, 
                theta
            )[:, 1:] + _lambda * x
        )
        x = x - update_x
        if np.max(abs(update_x)) < _tolerance:
            converged = True
    return theta, x
```

> It is possible to arrive at optimal \\(\theta\\) and \\(x\\) by repetitively minimizing them using \eqref{4} and \eqref{8}.

```python
tolerance=0.001
max_k=50

# the order of application of the estimate_x and estimate_theta can be altered
theta, x = estimate_x_v2(y, _tolerance=tolerance, max_k=max_k)

# iterating twice. more iterations would result in better convergence
for _ in range(2):
    theta, x = estimate_theta_v2(y, x=x, theta=theta, _tolerance=tolerance, max_k=max_k)
    theta, x = estimate_x_v2(y, x=x, theta=theta, _tolerance=tolerance, max_k=max_k)

predictions = np.matmul(np.hstack((np.ones((10, 1)), x)), theta.transpose())
```

But it is also possible to solve for both \\(\theta\\) and \\(x\\) simultaneously, given by an update rule which is nothing but the combination of the earlier two update rules in \eqref{3} and \eqref{7}. So the resulting cost function is given by,

$$
\begin{align}
J(x^{(1)}, \cdots, x^{(n_m)}, \theta^{(1)}, \cdots, \theta^{(n_u)}) &= {1 \over 2} \sum_{(i,j):r(i,j)=1} ((\theta^{(j)})^T x^{(i)} - y^{(i, j)})^2 \\ 
&+ {\lambda \over 2} \sum_{i=1}^{n_m} \sum_{k=1}^{n} (x_k^{(i)})^2 \\
&+ {\lambda \over 2} \sum_{j=1}^{n_u} \sum_{k=1}^n (\theta_k^{(j)})^2
\end{align}
\label{9} \tag{9}
$$

and the minimization objective can be written as,

$$
min_{x^{(1)}, \cdots, x^{(n_m)}, \theta^{(1)}, \cdots, \theta^{(n_u)}} J(x^{(1)}, \cdots, x^{(n_m)}, \theta^{(1)}, \cdots, \theta^{(n_u)}) \tag{10} \label{10}
$$

Practically, the minimization objective \eqref{10} is equivalent to \eqref{4} if \\(x\\) is kept constant. Similarly, it's equivalent to \eqref{8} if \\(\theta\\) is kept constant.

> In \eqref{10}, by convention there is no \\(x_0=1\\) and thus consequently, there in no \\(\theta_0\\), hence leading to \\(x \in \mathbb{R}^n\\) and \\(\theta \in \mathbb{R}^n\\). 

To summarize, the collaborative filtering algorithm has the following steps,

* Initializa \\(x^{(1)}, \cdots, x^{(n_m)}, \theta^{(1)}, \cdots, \theta^{(n_u)}\\) to small random values.
* Minimize \eqref{9} using gradient descent or any other advance optimization algorithm. The update rules given below can be obtained by following the partial derivatives along \\(x's\\) and \\(\theta's\\).

$$
\begin{align}
x_k^{(i)} &= x_k^{(i)} - \alpha \left( \sum_{j:r(i,j)=1} ((\theta^{(j)})^T x^{(i)} - y(i,j)) \theta_k^{(j)} + \lambda x_k^{(i)} \right) \\
\theta_k^{(j)} &= \theta_k^{(j)} - \alpha \left( \sum_{i: r(i, j)=1} ((\theta^{(j)})^T x^{(i)} - y^{(i, j)}) x_k^{(i)} + \lambda \theta_k^{(j)}\right)
\end{align} \tag{11} \label{11}
$$

```python
def colaborative_filtering_v2(y, max_k=2,
             _alpha=0.01, _lambda=0.001, _tolerance=0.001, r=None):
    if type(r) != np.ndarray:
        r = np.where(y>0, 1, 0)
    converged = False
    max_i, max_j = y.shape
    x = np.random.rand(max_i, max_k)
    theta = np.random.rand(max_j, max_k)
    
    while not converged:
        update_x = np.zeros(x.shape)
        update_theta = np.zeros(theta.shape)
        update_x = _alpha * (
            np.matmul(
                (np.matmul(x, theta.transpose()) - y) * r, 
                theta
            ) + _lambda * x
        )
        update_theta = _alpha * (
            np.matmul(
                x.transpose(),
                (np.matmul(x, theta.transpose()) - y) * r, 
            ).transpose() + _lambda * theta
        )
        x = x - update_x
        theta = theta - update_theta
        if max(np.max(abs(update_x)), np.max(abs(update_theta))) < _tolerance:
            converged = True
    return theta, x

theta, x = colaborative_filtering_v2(y, max_k=max_k)
predictions = np.matmul(x, theta.transpose())
```

* For a user with parameter \\(\theta\\) and a choice with learned features \\(x\\), the predicted star rating is given by \\(\theta^T x\\).

Consequently, the matrix of ratings, \\(Y\\), can be written as,

$$
Y = \left[ 
\begin{matrix}
(\theta^{(1)})^T x^{(1)} & (\theta^{(2)})^T x^{(1)} & \cdots & (\theta^{(n_u)})^T x^{(1)} \\
(\theta^{(1)})^T x^{(2)} & (\theta^{(2)})^T x^{(2)} & \cdots & (\theta^{(n_u)})^T x^{(2)} \\
\vdots & \vdots & \ddots & \vdots \\
(\theta^{(1)})^T x^{(n_m)} & (\theta^{(2)})^T x^{(n_m)} & \cdots & (\theta^{(n_u)})^T x^{(n_m)} \\
\end{matrix}
\right ] \tag{12} \label{12}
$$

Where \\(y(i, j)\\) is the rating for choice \\(i\\) by user \\(j\\).

Vectorized implementation of \eqref{12}, is given by,

$$Y = X \Theta^T \tag{13} \label{13}$$

Where, 

* each row \\(i\\) in \\(X\\) represents the feature vector of choice \\(i\\).
* each row \\(j\\) in \\(\Theta\\) represents the parameter vector for user \\(j\\).

> The algorithm discussed is also called low rank matrix factorization which is a property of the matrix \\(Y\\) is linear algebra.

### Similar Recommendations

After the collaborative filtering algorithm has converged, it can be used to find related choices. For each choice \\(i\\), a feature vector is learned, \\(x^{(i)} \in \mathbb{R}^n\\). Although it is generally not possible to decipher what the values in the matrix \\(X\\) denote, they encode representative features of the choices in detail. So in order to find choices close to a given choice \\(i\\), a simple euclidean distance calculation will give the desired results

> If the distance between choices \\(i\\) and \\(j\\) is small, i.e. \\(\lVert x^{(i)} - x^{(j)} \rVert\\) is small, then they are similar.

### Mean Normalization

Consider a case where one of the users has not rated any of the choices, then the rating matrix Y can be defined as,

```python
y = np.hstack((y, np.zeros((y.shape[0], 1))))
r = np.where(y > 0, 1, 0)
```

Since none of the choices are rated by this user, the entries in R matrix corresponding to this user would be all zeros. So, \eqref{9} can be written as follows (because \\({1 \over 2} \sum_{(i,j):r(i,j)=1} ((\theta^{(j)})^T x^{(i)} - y^{(i, j)})^2 = 0\\)), 

$$
J(x^{(1)}, \cdots, x^{(n_m)}, \theta^{(1)}, \cdots, \theta^{(n_u)}) = {\lambda \over 2} \sum_{i=1}^{n_m} \sum_{k=1}^{n} (x_k^{(i)})^2 + {\lambda \over 2} \sum_{j=1}^{n_u} \sum_{k=1}^n (\theta_k^{(j)})^2 \tag{14} \label{14}
$$

Since the updates to \\(\theta\\) corresponding to this user is only governed by this cost function, it would only minimize parameter vector \\(\theta\\). This can be seen easily by setting a low tolerance for the collaborative filtering,

```python
max_k = 5
tolerance = 0.0000001
theta, x = colaborative_filtering_v2(y, max_k=max_k, _tolerance=tolerance)
predictions = np.matmul(x, theta.transpose())
```

Obviously this is not a correct assumption to rate all the choices 0 for a user that has rated none so far. For such a user it would be ideal to predict the rating as the average of ratings attibuted to it by other users so far.

Mean normalization helps in acheiving this. In this process each row of the ratings matrix is normalized by its mean and later denormalized after predictions.

```python
def normalized(y, max_k=2,
             _alpha=0.01, _lambda=0.001, _tolerance=0.001):
    r = np.where(y>0, 1, 0)
    y_sum = y.sum(axis=1)
    r_sum = r.sum(axis=1)
    y_mean = np.atleast_2d(y_sum/r_sum).transpose()
    y_norm = y - y_mean
    theta, x = colaborative_filtering_v2(y_norm, max_k, _alpha, _lambda, _tolerance, r)
    return theta, x, y_mean

theta, x, y_mean = normalized(y, max_k=max_k, _tolerance=tolerance)
predictions = np.matmul(x, theta.transpose()) + y_mean
```

## REFERENCES:

<small>[Machine Learning: Coursera - Content Based Recommendations](https://www.coursera.org/learn/machine-learning/lecture/uG59z/content-based-recommendations){:target="_blank"}</small><br>
<small>[Machine Learning: Coursera - Collaborative Filtering](https://www.coursera.org/learn/machine-learning/lecture/2WoBV/collaborative-filtering){:target="_blank"}</small><br>
<small>[Machine Learning: Coursera - Algorithm](https://www.coursera.org/learn/machine-learning/lecture/f26nH/collaborative-filtering-algorithm){:target="_blank"}</small><br>
<small>[Machine Learning: Coursera - Low Rank Matrix Factorization](https://www.coursera.org/learn/machine-learning/lecture/CEXN0/vectorization-low-rank-matrix-factorization){:target="_blank"}</small><br>
<small>[Machine Learning: Coursera - Mean Normalization](https://www.coursera.org/learn/machine-learning/lecture/Adk8G/implementational-detail-mean-normalization){:target="_blank"}</small>

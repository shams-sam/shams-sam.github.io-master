---
layout: post
title: "K-Means Clustering"
categories: [basics-of-machine-learning]
tags: [machine-learning, andrew-ng]
description: K-means clustering, a method from vector quantization, aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster.
cover: "/assets/images/clustering.png"
cover_source: "http://graphalchemist.github.io/Alchemy/images/features/cluster_team.png"
comments: true
mathjax: true
---

{% include collection-basics-of-machine-learning.md %}

### Introduction

* K-means clustering is one of the most popular clustering algorithms.
* It gets it name based on its property that it tries to find most optimal user specified k number of clusters in a any dataset. The quality of the dataset and their seperability is subject to implementation details, but it is fairly straight forward iterative algorithm.
* It basically involves a random centroid initialization step followed by two steps, namely, cluster assignment step, and centroid calculation step that are executed iteratively until a stable mean set is arrived upon. It becomes more clear in the animation below.

![Fig-1 K-Means Animation](/assets/2018-04-19-k-means-clustering/fig-1-clustering-animation.gif?raw=true){:width="70%"}

* **Cluster Assignment**: Assign each data point to one of the two clusters based on its distance from them. A point is assigned to the cluster, whose centroid it is closer to.
* **Move Centroid:** After cluster assignment, centroids are moved to the mean of clusters formed. And then the process is repeated. After a certain number of steps the centroids will no longer move around and then the iterations can stop.

### K-Means Algorithm

Input:

- \\(K\\), number of clusters
- Training set, \\(x^{(1)}, x^{(2)}, \cdots, x^{(m)}\\)

where:

- \\(x^{(i)} \in \mathbb{R}^n\\), as there are no bias terms, \\(x_0=1\\)

Algorithm:

- Randomly initialize \\(K\\) cluster centroids, \\(\mu_1, \mu_2, \cdots, \mu_k \in \mathbb{R}^n\\)
- Then,

$$
\begin{align}
Repeat \{ \\
    \text{for }i &= 1\text{ to }m \\
        & c^{(i)} =\text{ index (from }1\text{ to }K\text{) of centroid closest to }x^{(i)}\text{, i.e., } min_k \lVert x^{(i)} - \mu_k \rVert^2 \\
    \text{for }k &= 1\text{ to }K \\
        & \mu_k =\text{ average (mean) of points assigned to cluster, }k\text{, i.e., } \frac {\text{sum of } x^{(i)}\text{, where }c^{(i)} = k} {\text{number of }c^{(i)} = k} \\
\}
\end{align}
\tag{1} \label{1}
$$

> It is common to apply k-means to a non-seperated clusters. This has particular applications in segmentation problems, like market segmentation or population division based on pre-selected features.

### Optimization Objective

Notation:

* \\(c^{(i)}\\) - index of cluster \\(\\{1, 2, \cdots, K\\}\\) to which example \\(x^{(i)}\\) is currently assigned
* \\(\mu_k\\) - cluster centroid \\(k\\), \\(\mu_k \in \mathbb{R}^n\\)
* \\(\mu_{c^{(i)}}\\) - cluster centroid of the cluster to which the example \\(x^{(i)}\\) is assigned

Following the above notation, the cost function of the k-means clustering is given by,

$$J(c^{(1)}, c^{(2)}, \cdots, c^{(m)}, \mu_1, \mu_2, \cdots, \mu_K) = {1 \over m} \sum_{i=1}^m \lVert x^{(i)} - \mu_{c^{(i)}} \rVert^2 \tag{2} \label{2}$$

Hence the optimization objective is,

$$min_{c^{(1)}, c^{(2)}, \cdots, c^{(m)},\\ \mu_1, \mu_2, \cdots, \mu_K} J(c^{(1)}, c^{(2)}, \cdots, c^{(m)}, \mu_1, \mu_2, \cdots, \mu_K) \tag{3} \label{3} $$

> The cost function in \eqref{2} is called distortion cost function or the distortion of k-means clustering.

It can argued that the k-means algorithm in \eqref{1}, is implementing the cost function optimization. This is so because the first step of k-mean clustering, i.e. the cluster assignment step is nothing but the minimization of the cost w.r.t. \\(c^{(1)}, c^{(2)}, \cdots, c^{(m)}\\) as this step involves assigning a data point to the nearest possible cluster. Similarly the second step, i.e. moving the centroid step is the minimization of the clustering cost w.r.t. \\(\mu_1, \mu_2, \cdots, \mu_K\\) as the most optimal position of centroid for minimizing the distortion for a given set of points is the mean position.

One handy way of checking if the clustering alorithm is working correctly is to plot distortion as a function of number of iterations. As both the steps in the k-means are calculated steps for minimization it is always going to decrease or remain approximately constant as the number of iterations increase.

### Random Initialization

There are various ways for randomly picking out \\(K < m\\) cluster centroids, but the most recommended one involves picking \\(K\\) randomly picked training examples and initialize \\(\\{\mu_1, \mu_2, \cdots, \mu_K\\}\\) equal to these \\(K\\) examples.

Based on initialization, it is possible that k-means could converge to different centroids or stuck in some local optima. One possible solution to this is to try multiple random initializations and then choose the one with the least distortion. It's fairly usual to run k-means around 50-1000 times with random initialization to make sure that it does not get stuck in local optima.

Generally the trick of multiple random initializations will help only if the number of clusters is small, i.e. between 2-10. For higher number of clusters the multiple number of random initializations are less likely to help improve the distortion cost function.

### Choosing the Number of Clusters

* One way of choosing the number of clusters is by manually visualizing the data.
* Sometimes it is ambiguous as to how many clusters exist in the dataset and in such cases it's rather more useful to choose the number of clusters on the basis of end goal or the number of clusters that serve well the later down stream goal that needs to be extracted from the datasets.
* **Elbow Method:** On plotting the distortion as a function of number of clusters, \\(K\\), this methods says that the optimal number of cluster at the point the elbow occurs as can be seen for line B in the plot below. It is a reasonable way of choosing the number of clusters. But this method does not always work because the sometimes the plot would look like line A which does not have clear elbow to choose.

![Fig-2 Elbow Method](/assets/2018-04-19-k-means-clustering/fig-2-elbow-method.png?raw=true){:width="70%"}

> As a strict rule in k-means, as the number of cluster \\(K\\) increases, the distortion would decrease. But after some point the increase in cluster would not give much decrease in the distortion.

### Example

[**Ipython Notebook**](https://github.com/shams-sam/CourseraMachineLearningAndrewNg/blob/master/k-means.ipynb){:target="\_blank"}

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np
```

Given a image it is reshaped into a vector for ease of processing, 

```python
l, w, ch = img.shape
vec_img = img.reshape(-1, ch).astype(int)
```

Following this K points are randomly chosen and assigned as centroids,

```python
def choose_random(K, vec):
    m = len(vec)
    idx = np.random.randint(0, m, K)
    return vec[idx]

mu = choose_random(K, vec_img)
```

The two basic steps of k-means clustering, cluster assignment and moving centroids can be implemented as follows,

```python
def cluster_assignment(mu, vec):
    return ((vec - mu[:, np.newaxis]) ** 2).sum(axis=2).argmin(axis=0)

def move_centroid(mu, c, vec):
    for i in range(len(mu)):
        vec_sub = vec[c==i]
        mu[i] = np.mean(vec_sub, axis=0)
    return mu
```

The distortion cost fuction is calculated as follows,

```python
def distortion(mu, c, vec):
    return ((mu[c] - vec) ** 2).sum() / vec.shape[0]
```

Once all the modules are in place, k-means needs to iterate over the steps of cluster assignment and moving centroids until the distorion is within the threshold (threshold chosen = 1),

```python
last_dist = distortion(mu, c, vec_img) + 100
curr_dist = last_dist - 100

while last_dist - curr_dist > 1:
    last_dist = curr_dist
    c = cluster_assignment(mu, vec_img)
    mu = move_centroid(mu, c, vec_img)    
    curr_dist = distortion(mu, c, vec_img)
```

Following plots are obtained after running k-means for image compression on two different images,

![K-Means Compression - Image 1](/assets/2018-04-19-k-means-clustering/fig-3-image-compression-1.png?raw=true){:width="70%"}

![K-Means Compression - Image 2](/assets/2018-04-19-k-means-clustering/fig-4-image-compression-2.png?raw=true){:width="70%"}

Following code implements k-means for different values of K.

```python
def elbow(img):
    K_hist = []
    dist_hist = []
    for K in tqdm(range(1, 10)):
        K_hist.append(K)
        mu, c, dist = k_means(img, K, plot=False)
        dist_hist.append(dist)
    plt.plot(K_hist, dist_hist)
    plt.xlabel("K")
    plt.ylabel("final distortion")

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.title('elbow plot of image 1')
elbow(img_1)
plt.subplot(1, 2, 2)
elbow(img_2)
plt.title('elbow plot of image 2')
```

![K-Means Compression - Image 2](/assets/2018-04-19-k-means-clustering/fig-5-elbow-plot.png?raw=true){:width="80%"}

Seeing the two plots it is evident while the elbow plot gives a optimal value of two for image 1, there is no well defined elbow for the image 2 and it is not very clear which value would be optimal as mentioned in the [section](#choosing-the-number-of-clusters) above.

## REFERENCES:

<small>[Machine Learning: Coursera - K-Means Clustering](https://www.coursera.org/learn/machine-learning/lecture/93VPG/k-means-algorithm){:target="_blank"}</small><br>
<small>[Machine Learning: Coursera - Optimization Objective](https://www.coursera.org/learn/machine-learning/lecture/G6QWt/optimization-objective){:target="_blank"}</small><br>
<small>[Machine Learning: Coursera - Random Initialization](https://www.coursera.org/learn/machine-learning/lecture/drcBh/random-initialization){:target="_blank"}</small><br>
<small>[Machine Learning: Coursera - Choosing the number of clusters](https://www.coursera.org/learn/machine-learning/lecture/Ks0E9/choosing-the-number-of-clusters){:target="_blank"}</small><br>
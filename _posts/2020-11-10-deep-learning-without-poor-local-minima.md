---
layout: post
title: Deep Learning without Poor Local Minima
categories: []
tags: [machine-learning, mathematics, papers, theorems]
description: This paper proves a conjecture about deep neural networks published in 1989 and addresses an open problem announced at COLT 2015.
cover: "/assets/images/nips-2016.png"
cover_source: "https://www.incimages.com/uploaded_files/image/1940x900/getty_468867139_157971.jpg"
comments: true
mathjax: true
---

### Importance

It has long been an open question to understand why is training a deep neural network tractable. Specifically, we know that deep neural networks are non-convex functions with possible local minima. We also know that stochastic gradient descent is prone to getting stuck in such local minima. What has confounded researchers for a long time is the fact that **while searching such a huge solution space, why does a neural network not get stuck in local minima" more often** and learn bad solutions. This work tries to give some understanding about these open questions.

### Prior works

There has an earlier paper in the field published back in 1989, that proves similar statements but under much tighter assumptions. This work builds on top of that using different techniques although and relaxing the assumptions and using only 2 or the original 7 used in the prior work.

### Interesting Results

Part of the main result says "Every critical point that is not a global minimum is a saddle point", which is interesting because it can be loosely translated to "there are no local maxima" which can be quite unintuitive to imagine at first.

> There are no local maxima.

### Prerequisites

A brief understanding of the following topics would be good to have:

- Linear Algebra. duh!
- First order and second order necessary conditions for local minima.
- Familiarity with graph interpretation of neural networks.

### Contributions

This paper proves the following statements for squared loss function of deep linear networks with any depth and any widths:

- The function is non-convex and non-concave.
- Every local minimum is a global minimum.
- Every critical point that is not a global minimum is a saddle point.
- Bad saddle points exist for deeper networks s.t. Hessian has no negative eigenvalue

Shallow networks don't have this issue of bad saddle points. This work further proves the same 4 statements via reduction about deep non-linear neural networks

## REFERENCES:

<small>[Deep Learning without Poor Local Minima](https://papers.nips.cc/paper/2016/hash/f2fc990265c712c49d51a18a32b39f0c-Abstract.html){:target="_blank"}</small>

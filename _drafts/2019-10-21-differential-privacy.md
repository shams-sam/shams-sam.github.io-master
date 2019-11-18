---
layout: post
title: What is Differential Privacy?
categories: []
tags: [mathematics, papers]
description: Differential Privacy is a meaningful and mathematically rigorous definition of of privacy useful for quantifying and bounding privacy loss. 
cover: "/assets/images/differential_privacy.png"
cover_source: "https://www.incimages.com/uploaded_files/image/1940x900/getty_468867139_157971.jpg"
comments: true
mathjax: true
---

### Introduction

- The method has been developed in context of **statistical disclosure control** - providing accurate statistical information about a set of respondents while protecting the privacy of each individual - and applies more generally to any private data set for which it is desirable to release coarse-grained information while keeping private the details.

- This basically means that DP will ensure a similar probability distribution over the cluster of data points, **independent of whether any individual opts in or out of the data set**.

- Formally, consider a pair of datasets $$D$$, $$D'$$ differing in atmost one row, i.e. one is subset of the other and the larger one has exactly one more row in it.

> Definition: A randomized function $$\mathcal{K}$$ gives $$\epsilon$$-differential privacy if for all data sets $$D$$ and $$D'$$ differing on at most one row and all $$S \subseteq Range(\mathcal{K})$$,
> 
> $$Pr[\mathcal{K}(D) \in S] \leq exp(\epsilon) \cdot Pr[\mathcal{K}(D') \in S]$$

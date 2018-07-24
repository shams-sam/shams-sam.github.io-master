---
layout: post
title: "Introduction to Survival Analysis"
categories: []
tags: [machine-learning, papers, mathematics]
description: The term survival time is used to describe the length of time until a specified event. The widespread use of these models in medicine to analyze survival times leads to the name survival analysis.
cover: "/assets/images/ecg.jpg"
cover_source: "https://images3.alphacoders.com/170/thumb-1920-170789.jpg"
comments: true
mathjax: true
---

### Introduction

Survival analysis refers to the set of statistical analyses that are used to analyze the length of time until an event of interest occurs. These methods have been traditionally used in analysing the survival times of patients and hence the name. But they also have a utility in a lot of different application including but not limited to analysis of the time of recidivism, failure of equipments, survival time of patients etc. Hence, simply put the phrase **survival time** is used to refer to the type of variable of interest. It is often also referred by names such as **failure time** and **waiting time**.

Such studies generally work with **data leading upto an event of interest** along with several other characteristics of individual data points that may be used to explain the survival times statistically.

The statistical problem (survival analysis) is to construct and estimate an appropriate model of the time of event occurance. A survival model fulfills the following expectations:

* yield predictions of number of individuals who will fail (undergo the event of interest) at any length of time since the beginning of observation (or other decided point in time).
* estimate the effect of observable individual characteristics on the survival time (to check the relevance of one variable holding constant all others).

It is often observed that the survival models such as proportional hazard model are capable of **explaining the survival times in terms of observed characteristics** which is better than straight-forward statistical inferences such as **rates of event occurence without considering characteristic features** of data.

### Basics

Assume **survival time T is a random variable** following some distribution **characterized by cumulative distribution function \\(F(t, \theta)\\)**, where 

* \\(\theta\\) is the set of **parameters to be estimated**
* \\(F(t, \theta) = P(T \leq t) = \\) **probability that there is a failure** at or before time \\(t\\), for any \\(t \geq 0\\)
* \\(F(t, \theta) \to 1\\) as \\(t \to \infty\\), since \\(F(t, \theta)\\) is a **cumulative distribution function**
* Above tendency leads to an **implicit assumption that all candidates would eventuall fail**. While this assumptions works selectively based on settings (true for patient survival times, not true for time of repayment of loans)and hence needs to be relaxed where it does not hold true.

**Survival times are non-negative** by definition and hence the distributions (like exponential, Weibull, gamma, lognormal etc.) characterising it are defined for value of time \\(t\\) from \\(0\\) to \\(\infty\\). 

Let \\(f(t, \theta\\)\\) be the **density function** correponding to the distribution function \\(F(t, \theta)\\), then the **survival function** is given by,

$$S(t, \theta) = 1 - F(t, \theta) = P(T \gt t) \tag{1} \label{1}$$

which gives the **probability of survival** until time \\(t\\) (\\(S(t, \theta) \to 0\\) as \\(t \to \infty\\) because, \\(F(t, \theta) \to 1\\) as \\(t \to \infty\\)).

Another useful concept in survival analysis is called **hazard rate**, defined by,

$$h(t, \theta) = \frac{f(t, \theta)} {1-F(t, \theta)} = \frac{f(t, \theta)} {S(t, \theta)} \tag{2} \label{2}$$

> Hazard rate represents the density of a failure at time \\(t\\), conditional on no failure prior to time \\(t\\), i.e., it indicates the probability of failure in the next unit of time, given that no failure has occured yet.

**While \\(f(t, \theta)\\) roughly represents the proportion of original cohort that should be expected to fail between time \\(t\\) and \\(t+1\\), hazard rate \\(h(t, \theta)\\) represents the proportion of survivors until time \\(t\\) that should be expected to fail in the same time window, \\(t\\) to \\(t+1\\).**


## REFERENCES:

<small>[Survival Analysis: A Survey](https://link.springer.com/article/10.1007/BF01083132#){:target="_blank"}</small><br>
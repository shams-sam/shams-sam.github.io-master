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

The relationship betwee the cumulative distribution function and the hazard rate is given by,

$$F(t, \theta) = 1 - exp \left[ - \int_0^t h(x, \theta) dx \right] \tag{3} \label{3}$$

and 

$$h(t, \theta) = - \frac {d\,ln\,[1 - F(t, \theta)]} {dt} \tag{4} \label{4}$$

The fact that \\(F(t, \theta)\\) is a cdf puts some restrictions on the hazard rate, 

- hazard rate is non-negative function 

$$H(t, \theta) = \int_0^t h(x, \theta) dx \tag{5} \label{5}$$

- the integrated hazard in \eqref{5} is finite for finite \\(t\\) and tends to \\(\infty\\) as \\(t\\) approaches \\(\infty\\).

### State Dependence

- Positive state dependence or an increasing hazard rate \\(dh(t)/dt \gt 0 \\) indicates that the **probability of failure during the next time unit increases** as the length of time at risk increases.
- Negative state dependence or a decreasing hazard rate \\(dh(t)/dt \lt 0 \\) indicates that the **probability of failure in the next time unit decreases** as the length of time at risk decreases.
- No state dependence indicates a **constant hazard rate**.

> Only exponential distribution displays no state dependence.

### Censoring and Truncation

A common feature of data on survival times is that they are censored or truncated. Censoring and truncation are statistical terms that refer to the **inability to observe the variable of interest for the entire population**.

- A standard example to understand this can be understood in the form of a case of an individual shooting at a round target with a rifle and the variable of interest is the distance by which the bullet misses the center of the target. 
- If all shots hit the target, this distance can be measure for all the shots and there is no problem of censoring or truncation.
- If some shots miss the target, but we know the number of shots fired, **the sample is censored**. In this case either the distance of shot from center is known or it is known that it was atleast as large as the radius of the target.
- Similarly if one does not know how many shots were fired but only have information about distance for shots that hit the target, **the sample is truncated**.

> Censored sample has more information than a truncated sample.

Survival times are often censored because not all candidates would fail by the end of time during which the data was collected. This **censoring of data must be taken into account** while making the estimations because it is **not legitimate to drop such observations** with unobserved survival times **r to set survival times for these observations equal to the length of the follow-up period** (when the data was collected).

- Infrequently so, but there is also a chance of getting information about a candidate during a follow-up collection who was not a part of the original population. In such cases the **survival time is truncated** because there is no information of the candidate or his survival time.

### Problem of Estimation

The initial assumption specifies a cumulative distribution function \\(F(t, \theta)\\), or equivalently a density \\(f(t, \theta)\\) or hazard \\(h(t, \theta)\\) that is of a known form except that it depends on a unknown parameter \\(\theta\\). Estimation of this parameter is first step for the model to make any meaningful prediction about the survival time of new candidate

Consider a case of estimation of parameter for a censored sample which is defined as follows,

- sample has \\(N\\) individuals with follow-up periods \\(T_1, T_2, \cdots, T_N\\). These follow-ups may be all equal, but they usually are not.
- \\(n\\) is number of individuals who fail, numbered \\(1, 2, \cdots, n\\) and individuals numbered \\(n+1, n+2, \cdots, N\\) are the non-failures.
- for the candidates who fail, there exists a survival time \\(t_i \leq T_i, \, i \in [1, n]\\) 
- for the non-failures, survival time \\(t_i\\) is not observed but it is known that it is greater than the length of the follow-up period \\(T_i\\), \\(i \in [n+1, N]\\).

If it is assumed that **all the outcomes are independent** of each other the likelyhood function of the sample is, 

$$L = \prod_{i=1}^n f(t_i, \theta) \prod_{i=n+1}^N S(T_i, \theta) \tag{6} \label{6}$$

> Likelyhood function is a general statistical tool that expresses the probability of outcomes observed in terms of unknown parameters that are to be estimated, i.e., it is function of the parameters to be estimated, which serves as a measure of how likely it is that the statistical model, with a given parameter value, would generate the given data.

A common used estimator of \\(\theta\\) is the **Maximum Likelyhood Estimator (MLE)** which is defined as the value of \\(\theta\\) that maximizes the likelyhood function.

## REFERENCES:

<small>[Survival Analysis: A Survey](https://link.springer.com/article/10.1007/BF01083132#){:target="_blank"}</small><br>
---
layout: post
title: "Introduction to Survival Analysis"
categories: []
tags: [machine-learning, papers, featured]
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
* Above tendency leads to an **implicit assumption that all candidates would eventually fail**. While this assumptions works selectively based on settings (true for patient survival times, not true for time of repayment of loans) and hence needs to be relaxed where it does not hold true.

**Survival times are non-negative** by definition and hence the distributions (like exponential, Weibull, gamma, lognormal etc.) characterising it are defined for value of time \\(t\\) from \\(0\\) to \\(\infty\\). 

Let \\(f(t, \theta)\\) be the **density function** correponding to the distribution function \\(F(t, \theta)\\), then the **survival function** is given by,

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

The MLE have been shown to display the following desirable properties over a large sample (**as the sample size approaches infinity**),

- Unbiased
- Efficient
- Normally Distributed

As mentioned, the properties of MLE are only **relevant when the sample size is large**. It is often **observed that the sample sizes in these studies are much smaller** and hence reliance on large sample properties of estimator is more tenuous.

The above survival model uses observed survival time \\(t_i\\) while **ignoring the specific timing of the observed returns**. So the analysis of the fact of failure or non-failure, ignoring the timing of observed failures, would properly be based on the likelyhood function 

$$L = \prod_{i=1}^n F(T_i, \theta) \prod_{i=n+1}^N S(T_i, \theta) \tag{7} \label{7}$$

Estimation using \eqref{7} is a legitimate procedure and does not cause any bias or inconsistency, but the estimates are inefficient relative to MLEs from \eqref{6}.

**The estimates of \\(\theta\\) gotten by maximizing \eqref{7} will be less efficient (have larger variance) than the estimates of \\(\theta\\) gotten by maximizing \eqref{6}, atleast for large sample sizes. Hence if the information on time of return is available, it should be used.**

If the **truncated** case of sample is considered, then there is **no information on all the individuals who do not fail**. Formally, one starts with a cohort of \\(N\\) candidates, where **\\(N\\) is unknown**, and the only **observations available are the survival times \\(t_i\\) for the \\(n\\) individuals who fail** before the end of follow-up period. The \\(n\\) individuals appear in sample because \\(t_i \leq T_i\\), and the appropriate density is therefore

$$f(t_i, \theta \mid t_i \leq T_i) = \frac{f(t_i, \theta)}{P(t_i \leq T_i)} =  \frac{f(t_i, \theta)}{F(T_i, \theta)} \tag{8} \label{8}$$

And the corresponding **likelyhood function** which can be maximized to obtain the MLEs is given by,

$$L = \prod_{i=1}^n f(t_i, \theta \mid t_i \leq T_i) = \prod_{i=1}^n \frac{f(t_i, \theta)}{F(T_i, \theta)} \tag{9} \label{9}$$

### Explanatory Variables

Information on **explanatory variables may or may not be used** in estimating survival time models. Some models that are based on the **implicit assumption that distribution of survival time is the same for all individuals**, do not use explanatory variables.

**But practically it is observed that some individuals are more prone to failing than others and hence if information on individual charestistics and environmental variables is available, it should be used.**

This information can be incorporated is survival models by letting the parameter \\(\theta\\) depend on these individual characteristics and a new set of parameters. E.g. exponential model depends on a single parameter, say \\(\theta\\), and **\\(\theta\\) can be assumed to depend on the individual characteristics** as in linear regression.

### Non-Parametric Hazard Rate and Kaplan Meier

Before beginning any formal analyses of the data, it is often instructive to check the hazard rate. For this purpose, the **time until failure are rounded to the nearest quantized time unit** (month, week, day etc.). Following this it is easy to count the **number of candidates at risk at the beginning of the said time period** (i.e. the number of individuals who have not yet failed or been censored at the beginining of the time unit) and the **number of individuals who fail during the time period**. 

Then, the **non-parametric hazard rate** can be estimated as the ratio of number of failures during the time period to the number of individuals at risk at the beginning of time period, i.e., if the number of individuals at risk at the beginning of time \\(t \, (t = 1, 2, \cdots)\\) is denoted by \\(r\\), and the number of individuals who fail during this time \\(t\\) is denoted by \\(n_t\\), then the estimated hazard for time \\(t\\), \\(\hat{h}(t)\\) is given by,

$$\hat{h}(t) = \frac{n_t}{r} \tag{10} \label{10}$$

Such estimated hazard rates are prone to high variability. Also this high variability makes the purely non parametric estimates unattractive as they are less likely to give an accurate prediction on a new dataset. The parametric models such as exponential, Weibull or lognormal take care of this high variability and makes the model more tractable.

**But the plots of non parametric estimates of hazard rate provides a good initial guide as to which probability distribution may work well for a given usecase.**

As noted earlier, the hazard function, density function, and distribution function are alternative but equivalent ways of characterizing the distribution of the time until failure. Hence, once the hazard rate is estimated, then implicitly so is the density and the distribution function. It is possible to solve explicitly for the estimated density of distribution function in terms of the estimated hazard function. The resulting estimator (called **Kaplan Meier** or **product limit** estimator in statistical literature which is nothing but the non-parametric estimate) of the distribution function is given by,

$$\hat{F}(t) = 1 - \prod_{j=1}^t [1 - \hat{h}(j)] \tag{11} \label{11}$$

### Models without Explanatory Variables

There are various models that do not consider the explanatory variables, and instead **assume some specific distribution** such as exponential, Weibull, or lognormal for the length of time until failure. Essentially, the distribution of time until failure is known, except for some **unknown parameters that have to be estimated**. Hence, models of this type are called parametric models, which are different from the models discussed before as the later have no associated parameters or distribution.

The unknown parameters are **estimated by maximizing the likelyhood function** of the form \eqref{6}. 

> In case of exponential distribution, MLEs cannot be written in closed form (i.e. expressed algebraically), and so the maximization of likelyhood function is done numerically.

Once the characteristic parameters have been estimated, one can determine the following (which cannot be determined in case of non-parametric estimates like Kaplan Meier):

- **mean time** until failures
- **proportion of population that should be expected to fail** within any arbitrary period of time.

While the **advantage of such models lies in the smoothness of predictions**, the **disadvatage is the fact that it can be wrong and inturn lead to statements that are systematically misleading**. 

**Exponential Distribution**

The exponential distribution has density,

$$f(t) = \theta \, e^{-\theta t} \tag{12} \label{12}$$

and **survivor function**,

$$S(t) = e^{-\theta t} \tag{13} \label{13}$$

where

- the parameter is constrained, \\(\theta \gt 0\\)
- mean: \\(1 / \theta\\) and variance: \\(1 / \theta^2\\)
- only distribution with a **constant hazard rate**, specifically \\(h(t) = \theta\\) for all \\(t \geq 0\\)
- such hazard rates are generally seen in some physical processes such as radioactive decay.
- it is often not the most reasonable distribution for survival models.
- exponential distribution requires estimation of single parameter \\(\theta\\).

Consider a sample of \\(N\\) individuals, of which \\(n\\) have failed before the end of the follow-up period. The observed failure times be denoted by \\(t_i\, (i=1, 2, \cdots, n)\\) and the censoring times (length of follow up) for the non-failures de denoted by \\(T_i\, (i = n+1, \cdots, N)\\). Then the likelyhood function \eqref{6} can be written as 

$$L = \prod_{i=1}^n \theta\,e^{-\theta t_i} \prod_{i=n+1}^N e^{-\theta T_i} \tag{14} \label{14}$$

Maximizing \eqref{14} w.r.t. \\(\theta\\) yields MLE in closed form:

$$\hat{\theta} = \frac {n} {\sum_{i=1}^n t_i + \sum_{i=n+1}^N T_i} \tag{15} \label{15}$$

For large samples \\(\hat{\theta}\\) is normal with mean \\(\theta\\) and variance

$$\frac{\theta^2}{\sum_{i=1}^N [1 - exp(-\theta T_i)]} \tag{16} \label{16}$$

which for large \\(N\\) is adequately approximated by \\(\theta^2/n\\).

- Exponential distribution is highly skewed.
- Mean may not be a good measure of central tendency for exponential distribution.
- Median may be more preferrable indicator in most cases.

> Logarithm of likelyhood or log-likelyhood is used as a value to measure the goodness of fit. A higher value (more positive or less negative) for this variable indicates that the model fits the data better.


**Weibull Distribution**

In statistical literature, a very common alternative to the exponential distribution is the Weibull distribution. It is a generalization of the exponential distribution. By using Weibull distribution one can test to check if a simpler exponential model is more appropriate.

- A variable \\(T\\) has Weibull distribution if \\(T^{\tau}\\) has an exponential distribution for some value of \\(\tau\\).
- increasing hazard rate if \\(\tau \gt 1\\) and decreasing hazard rate if \\(\tau \lt 1\\). Also, if \\(\tau = 1\\) the hazard rate is constant and the Weibull distribution reduces to the exponential.
- **Weibull distribution has a monotonic hazard rate**, i.e it can be increasing, constant or decreasing but it cannot be increasing at first and then decreasing after some point.

The density of Weibull distribution is given by,

$$f(t) = \tau \theta^{\tau} \, t^{\tau -1} e^{-(\theta t)^\tau} \tag{17} \label{17}$$

and the survivor function is,

$$S(t) = e^{-(\theta t)^\tau} \tag{18} \label{18}$$

The likelyhood function for Weibull distribution can be derived by substituting \eqref{17} and \eqref{18} in \eqref{6}.


**Lognormal Distribution**

If \\(z\\) is distributed as \\(N(\mu, \sigma^2)\\), then \\(y = e^z\\) has a lognormal distribution with mean

$$\phi = exp(\mu + {1 \over 2} \sigma^2) \tag{19} \label{19}$$

and variance,

$$\tau^2 = exp(2 \mu + \sigma^2) [exp(\sigma^2) -1] = \phi^2 \psi^2 \tag{20} \label{20}$$

where 

$$\psi^2 = exp(\sigma^2) - 1 \tag{21} \label{21}$$

The **density** of \\(z = ln \, y\\) is the density of \\(N(\mu, \sigma^2)\\) given by,

$$f(ln \, y) = (1 / \sqrt{2\pi} \sigma) exp [-(1/2 \sigma^2) (ln\, y - \mu)^2] \tag{22} \label{22}$$

Generally there is **no advantage to working with the density of \\(y\\) itself, rather than \\(ln \, y\\)**. Thus, one can simply assume that log of survival time is distributed normally, and hence the likelyhood function \eqref{6} becomes

$$
\begin{align}
L = &- {n \over 2} ln(2\pi) - {n \over 2} ln(\sigma^2) - {1 \over 2\sigma^2} \sum_{i=1}^n (ln\, t_i - \mu)^2 \\
&+ \sum_{i=n+1}^N ln \, F \left[ \frac {\mu - ln\, T_i} {\sigma} \right]
\end{align}
\tag{23} \label{23}
$$

- where **\\(F\\) is the cumulative distribution function** for \\(N(0, 1)\\) distribution.
- **No analytical solution** exists for the maximization of \eqref{23} w.r.t. \\(\mu\\), and \\(\sigma^2\\), so it **must be maximized numerically**.
- the hazard function for lognormal distribution is complicated; it **increases first and then decreases**.

**Other distributions**

Although exponential, Weibull and lognormal are among the three most used distributions, there are various other well-known probability distributions possible, such as 

- log-logistic
- LaGuerre
- distributions based on Box-Cox power transformation of the normal

There are various ways of measuring how well models fit the data:

- value of likelyhood (or log-likelyhood) function
- maximum difference between the fitted value and actual cumulative distribution function
- standard Kolmogorov-Smirnov test of goodness of fit
- chi-square goodness-of-fit statistic based on predicted and actual failure times.

Over time it has been observed that even though some of these parametric distributions **might fit the data** better than others and excel on various metrics of good fit of data, these **do not give any explaination about the reasons governing the distribution** or any **insight into the affecting parameters** that lead to the different survival times in a population. Hence, these parametric models without the explanatory variables are not considered to be an effective tool for analysis.

### Models with Explanatory Variables

- Explanatory variables are in general added to survival models in an attempt to make more accurate predictions: the practical experiments over time corroborate the fact that individual characteristics, previous experiences and environmental setup helps predict whether or not a person will fail.

- An analysis of survival time without using the explanatory variables amounts to an analysis of its **marginal distribution**, whereas an analysis using explanatory variable amounts to an analysis of the **distribution of survival time conditional on these variables**.

> Variance of the conditional distribution is less than the variance of the marginal distribution, i.e. expect more precise distribution from former.

- Another more fundamental reason may include the interest of understanding the effect of explanatory variables on the survival time.

- More generally, these variables might be the demographics or environmental characteristics.

### Proportional Hazards Model

- allows one to estimate the effects of individual characteristics on survival time without having to assume a specific parametric form of distribution of time until failure.

- For an individual with the vector of characteristics, \\(x\\), the proportional hazards model assumes a hazard rate of the form, 

$$h(t \mid x) = h_0(t) e^{x_i^\prime \beta} \tag{24} \label{24}$$

where \\(h_0(t)\\) is completely arbitrary and unspecified baseline hazard function. **Thus, the model assumes that the hazard functions of all individuals differ only by a factor of proportionality,** i.e. if an individuals hazard rate is 10 times higher than another's at a given point of time, then it must be 10 times higher at all points in time. **Each hazard function follows same pattern over time.** 

However, there is no restriction on what this pattern can be, i.e. it puts no restriction on the \\(h_0(t)\\) curve, which determines the shape of \\(h(t \vert x)\\) curve. **\\(\beta\\) can be estimated without specifying \\(h_0(t)\\), and \\(h_0(t)\\) can be estimated non-parametrically and thus with flexibility.**


Consider a sample of \\(N\\) individuals, \\(n\\) of whom fail before the end of their follow-up period. Let the observations be ordered such that individual 1 has the shortest failure time, individual 2 has the second shortest failure time, and so forth. Thus, for individual \\(i\\), failure time \\(t_i\\) is observed, with,

$$t_1 \lt t_2 \lt \cdots \lt t_n \tag{25} \label{25}$$

A vector \\(x_i\\) represents individual characteristics for each individual \\(i = 1, 2, \cdots, N\\), irrespective of whether they failed.

For each observed failure times, \\(t_i\\), \\(R(t_i)\\) is defined as set of all individuals who were at risk just prior to time \\(t_i\\), i.e., it includes the individuals with failure times greater than or equal to \\(t_i\\), as well as the individuals whose follow-up is at least of length \\(t_i\\).

Using these definitions, the **partial-likelihood** function proposed by Cox can be defined for any failure time \\(t_i\\), as the probability that it is individual \\(i\\) who fails, given that exactly one individual from set \\(R(t_i\\)) fails, is given by,

$$\frac {h(t_i \vert x_i)} {\sum_{j \in R(t_i)} h(t_i \vert x_j)} = \frac {exp(x_i^\prime \beta)} {\sum_{j \in R(t_i)} exp(x_j^\prime \beta)} \tag{26} \label{26}$$

The partial-likelyhood function is formed by multiplying \eqref{26} over all \\(n\\) failure times,

$$ L = \prod_{i=1}^n \frac {exp(x_i^\prime \beta)} {\sum_{j \in R(t_i)} exp(x_j^\prime \beta)} \tag{27} \label{27}$$

The estimate of \\(\beta\\) by maximizing \eqref{27} numerically w.r.t \\(\beta\\) is the **partial maximum-likelyhood estimate**. The word **partial** in partial likelyhood refers to the fact that not all available information is used in estimating \\(\beta\\), i.e., it only depends on knowing which individuals were at risk when each observed failure occured. The exact numerical values of the failure times \\(t_i\\) or of the censoring times for the non recedivists are not needed; only their **order matters**.

Once \\(\beta\\) is estimated, \\(h_0(t)\\), the baseline hazard function can be estimated non-parametrically. The estimated baseline hazard function is constant over the intervals between failure times. One can also calculate **survivor function** \\(S_0(t)\\) or equivalently the baseline cumulative distribution function \\(F_0(t)\\), that corresponds to the estimated baseline hazard function.

**The estimated survivor function is a step function that falls at each time at which there is a failure.**

The point of proportional hazard model is that the survivor function is estimated non-parametrically (i.e. not imposing any structure on its pattern over time, except that it must decrease as \\(t\\) increases) and estimation of \\(\beta\\) can proceed seperately from estimation of survivor function.

### Split Population Models

The models considered so far assume some cumulative distribution function, \\(F(t)\\) for the survival time, that gives the probability of a failure upto and including time \\(t\\), and it approaches one as \\(t\\) approaches infinity. This basically means that every individual must eventually fail, if they were observed for long enough time. This assumption is not true in all cases.

**Split Population Models** (or split models) do not imply that every individual would eventually fail. Rather the population is divided into two groups, one of which would never fail.

Mathematically, let \\(Y\\) be an observable indicator with two values, one implying ultimate failure and zero implying perpetual success. Then,

$$
\begin{align}
P(Y=1) &= \delta \\
P(Y=0) &= 1 - \delta
\end{align}
\tag{28} \label{28}
$$

where \\(\delta\\) is the proportion of the population that would eventually fail, and \\(1 - \delta\\) is the proportion that would never fail.

Let \\(g(t \vert Y=1)\\) be density of survival times for the ultimate failures, and \\(G(t \vert Y=1)\\) be the corresponding cumulative distribution function. If one considers exponential model to represent them, then

$$
\begin{align}
g(t \vert Y=1) = \theta e^{-\theta t} \\
G(t \vert Y=1) = 1 - e^{-\theta t}
\end{align}
\tag{29} \label{29}
$$

It can also be noted that \\(g (t \vert Y = 0)\\) and \\(G(t \vert Y=0)\\) are not defined.

Let \\(T\\) be the length of the follow up period and let \\(R\\) be an observable indicator equal to one if there is failure by time \\(T\\) and zero if there is not. The probability for individuals who do not fail during the follow up period, i.e, the event of \\(R = 0\\) is given by,

$$
\begin{align}
P(R=0) &= P(Y=0) + P(Y=1)P(t \gt T \vert Y=1) \\
&= 1 - \delta + \delta e^{-\theta T}
\end{align}

\tag{30} \label{30}
$$

Similarly, probability density for people who fail with survival time \\(t\\) is given by,

$$
P(Y=1)P(t \lt T \vert Y=1) g(t \vert t \lt T, Y=1) = P(Y=1) g(t \vert Y=1) = \delta \theta e^{-\theta t} \tag{31} \label{31}
$$

So the likelyhood function is made up of \eqref{29} for those who do not fail and \eqref{30} for those who do. It is given by,

$$
L = \prod_{i=1}^n \delta \theta exp(-\theta t_i) \prod_{i = n+1}^N (1 - \delta + \delta exp(-\theta T_i)) \tag{32} \label{32}
$$

The maximum likelyhood estimate of both \\(\theta\\) and \\(\delta\\) can be obtained by maximizing \eqref{32} numerically. It can be noted that when \\(\delta = 1\\), \eqref{32} reduces to \eqref{14}, the original exponential survival time model.

The split population model can be seen as a model of two seperate subpopulations, one with hazard rate \\(\theta\\) and other with zero. A more generalized model exists where the subpopulations exist with two non-zero hazard rates namely, \\(\theta_1\\) and \\(\theta_2\\). Such models help to account for population that is heterogenous in nature. 

Split models can also be based on other distributions such as lognormal etc. Also, it is possible to include explanatory variables into a split model. In such cases, the explanatory variables maybe taken to affect the probabiliy of failure, \\(\delta\\) or distribution of time until failure. 

For example, for a given feature vector \\(x_i\\) of explanatory variables, using **logit/individual lognormal model**, \\(\delta\\) is modeled using, 

$$\delta_i = 1/(1+exp(x_i^\prime \alpha)) \tag{33} \label{33}$$

and parameter \\(\mu\\) of the lognormal distribution is given by,

$$\mu_i = x_i^\prime \beta \tag{34} \label{34}$$

Here, the parameter \\(\alpha\\) gives the effect of \\(x_i\\) on the probablity of failure, and \\(\beta\\) gives the effect of \\(x_i\\) on the time until failure.

Such models are of importance because they let one distinguish between effects of explanatory variable on probability of eventual failure from effects on time until failure who eventually do fail.

### Heterogeneity and State Dependence

The two major causes of observed declining hazard rates are:

- state dependence
- heterogeneity

The phenomenon of an actually decreasing hazard rate over time due to an actual change in behavior over time at individual level is referred to as **state dependence**.

The second possible reason is **heterogeneity**. This basically means that the hazard rates are different across individuals, i.e., some individuals are more prone to failure than others. Naturally, individuals with higher hazard rates tend to fail earlier, on average, than individuals with lower hazard rates. As a result the average hazard rate of the surviving group will decrease with length of time simply because the most failure prone individuals have been removed already. This is true even without state dependence, i.e, each individual has a constant hazard rate but hazard rate varies across individuals. Even such a group would display decreasing hazard rate.

It is important to understand the difference because a decrease in a hazard rate due to state dependance means a success of the underlying program, while decrease due to heterogeneity does not imply that the program is effective in preventing failure, because it is happening by the virtue of the data at hand.

### Time Varying Covariates

Until now explanatory variables affecting the time until failure do not potray changing values over time, but is a possibility that can not be denied.

The types of explanatory variables can be categorizaed as follows:

- variables that do not change over time, e.g race, sex etc.
- variables that change over time but not within a single follow-up period, e.g. number of times followed up etc.
- variables that change continuously over time, such as age, education etc.

The last type of variables make it reasonable to use a statistical model that allows covariates to vary over time. Such incorporation is relatively straightforward in hazard-based models such as proportional hazard models. At each point in time, hazard rate is determined by the values of explanatory variables at that time.

However, it is much more difficult to introduce time-varying components into parametric models because these models are parameterized in terms of density and cumulative distribution function, and the density of distribution function at time \\(t\\) depends on the whole history of the explanatory variables up to time \\(t\\). **In the presence of time varying covariates, a parameterization of the hazard rate would be much more convenient.**

**Panel or Longitudinal Data:** data on individuals over time without reference to just a single follow-up. Such datasets include a large number of time-varying explanatory variables.

## REFERENCES:

<small>[Survival Analysis: A Survey](https://link.springer.com/article/10.1007/BF01083132#){:target="_blank"}</small><br>
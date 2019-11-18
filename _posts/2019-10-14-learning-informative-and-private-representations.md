---
layout: post
title: Privacy Preserving Predictive Modeling GANs
categories: []
tags: [GAN, machine-learning, papers, privacy-gans]
description: This paper examines a GAN architecture to generate private encodings by ensuring a three player min-max optimization that regulates information leakage.
cover: "/assets/images/privacy.jpg"
cover_source: "https://www.incimages.com/uploaded_files/image/1940x900/getty_468867139_157971.jpg"
comments: true
mathjax: true
---

### Introduction

In the current era of increasing data sharing and ubiquity of machine learning there is a very little focus on privacy of the subjects whose data is being shared at such large scale. While most data released promises anonymization of the PIIs, there is much work in literature to point of that simple anonymization techniques fail to mask users if there is an access to auxilliary dataset in which such features are not masked.

In order to fix this issue there has been study in the field of privacy preservation for the datasets in public domain to ensure public trust in sharing their information which eventually is the root to building amazing machine learning models and drawing insights from big data.

This paper discusses a new architecture of GAN which tries to achieve this very objective of anonymization of the private data but in a deep learning setting where the work in overseen by an objective function for the encoder which embeds in itself the notion of anonymizing sensitive columns while trying to maximize the predivity of non-sensitive data columns.

### About the paper

The GAN framework presented consists of three components (explained in detail in later sections):

- Encoder
- Ally that predicts the desired variables,
- Adversary that predicts the sensitive data.

The objective of the GAN framework is two-fold: 

- learn low-dimensional representation of data points (users in this case) that excels at  classifying a desired task (whether a user will answer quiz question correctly here)
- prevent an adversary from recovering sensitive data (each users identity in this case)

### Background 

Netflix dataset one of the famous datasets which is used for various starter tutorials on recommendation system. The history of privacy breach in this particular case is related to this seemly harmless data. While the company had ensured data anonymization to prevent breach of privacy. It was later discovered that it is easy to locate the users with good accuracy using auxilliary data from other related datasets such as IMDB which does not anonymize its dataset. Similarly there are cases in the domain of insurance companies where it has been proven that reverse engineering is a viable effort despite the anonymization.

It has similarly been shown that one can identify anonymous users in online social networks with upto 90% accuracy. These metrics point to the fact that it is possible for an attacker to uncover sensitive attributes from user data. 

One of the popular work that has caught traction in this field is called Differential Privacy (DP), which proposes to add random noise to raw data, where the noise (generally from Laplacian Distribution) level controls the trade off between predictive quality and user privacy. But it has been found that this mechanism also reduces the utility of the data for predictive modeling and increases the sample complexity 

The GAN model presented in this particular work is an effort to achieve the privacy (to prevent de-anonymization) while preserving the predictive aspects of the dataset (to overcome the drawbacks of techniques like DP).

### Contributions

The authors apply this GAN architecture in a online MOOC setting. The objectives of the work include:

- Use the student data to predict whether or not they will answer a quiz correctly.
- Ensure that the encoded data does not achieve a good convergence on sensitive data such as user identity.

![GAN Architecure](/assets/2019-10-14-learning-informative-and-private-representations/fig-1-gan-architecture.png?raw=true)

The work is different from DP in two key aspects:

- It is data-dependent, i.e., it learns representations from user data
- Directly uses raw user data without relying on feature engineering

The objective of the GAN is to generate representations that minimize the loss function of the ally while maximizing the loss function of adversary.

One key advantage mentioned about the architecture is that it is model agnostic, i.e. each module can instantiate a specific differential function (e.g. neural networks) based on the needs of the particular application

### Algorithm

![](/assets/2019-10-14-learning-informative-and-private-representations/fig-2-algorithm.png?raw=true)

### Datasets and Objectives

This particular paper presents the empirical results on dataset from the course Networks: Friends, Money and Bytes on Coursera MOOC platform. This has a total of 92 in-video quiz questions among 20 lectures. Each lecture has 4-5 videos. A total of 314,632 clickstreams were recorded for 3976 unique students. 

Two types of data are collected about students:
- Video-watching clickstream: behavior is recorded as a sequence of clickstreams based on actions available in the scrub bar.
- Question submissions: answers submitted by a student to an in-video question

The final **objective** is defined as a mapping from student's interaction (clickstream) on a video to their performance on questions (data acquired regarding question submissions)

The data collected can have both time-varying as well as static attributes. Time varying attributes include the series of clickstream before a question is answered, while the static attribute will included metrics like fraction of course completed, amount of time spent etc.

### Metrics

- Accuracy on binary prediction of questions answered
- AUC-ROC curve to assess the tradeoff between true and false positive rates
- K Ranks and Mean average precision at K (MAP@K) to measure performance of privacy preservation

### Baselines

- Only one baseline benchmark is included in the work which is Laplace Mechanism in DP (Differential Privacy) which simply adds Laplace noise to the data.

### Findings and Conclusions

- The new architecture outperforms DP in terms of prediction task on question answers. It actually performs slightly better than the original features themselves.
- With parameter $$\alpha \to 1$$ in the GAN architecture, encoder is biased towards prediction than sensitive data obfuscation which is theoretically correct.
- Larger $$\epsilon$$ in DP means adding smaller noise component to the actual data, and it can be seen that models are better at predictive performance under such a setting.
- Larger sizes of encoding dimension ensures more preserved information towards both prediction and sensitive data with identical $$\alpha$$ values. This confirms the fact that the size of representations controls the amount of information contained in data representation.
- Raw clickstream data with LSTM performs better than the hand-crafted features in terms of the tradeoff between prediction quality vs user privacy.

### Follow-up Citations

- J. Bennett, S. Lanning et al., “The netflix prize,” in Proceedings of KDD
cup and workshop, vol. 2007. New York, NY, USA, 2007, p. 35.
- A. Narayanan and V. Shmatikov, “Robust de-anonymization of large sparse datasets,” in Security and Privacy, 2008. SP 2008. IEEE Symposium on. IEEE, 2008, pp. 111–125.
- “De-anonymizing social networks,” in Security and Privacy, 2009
30th IEEE Symposium on. IEEE, 2009, pp. 173–187.
- Dwork, F. McSherry, K. Nissim, and A. Smith, “Calibrating noise
to sensitivity in private data analysis,” in Proc. Theory of Cryptography
Conference, Mar. 2006, pp. 265–284.
- “Calibrating noise to sensitivity in private data analysis,” in Theory of Cryptography Conference. Springer, 2006, pp. 265–284.
- C. Huang, P. Kairouzyz, X. Chen, S. L., and R. Rajagopal, “Context-aware generative adversarial privacy,” arXiv preprint arXiv:1710.09549, Dec. 2017.


# REFERENCES

<small>[Learning Informative and Private Representations via Generative Adversarial Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8622089&tag=1){:target="_blank"}</small>
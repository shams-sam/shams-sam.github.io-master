---
layout: post
title: Generative Adversarial Networks
categories: []
tags: [gan, machine-learning, papers, privacy-gans]
description: GAN originally presented by Goodfellow et al is a novel technique that uses a minmax two-player game to learn latent data distributions. 
cover: "/assets/images/gan.png"
cover_source: "https://user-images.githubusercontent.com/8406802/45344464-683fb800-b569-11e8-90fe-a228401ffadd.png"
comments: true
mathjax: true
---

### Introduction

The basic **adversarial** framework of the GAN architecture can be broken down into the following **two players**:
- A **generative** model $$G$$, that tries to capture the latent data distribution.
- A **discriminative** model $$D$$, that estimates the probability that a sample came from training data rather than $$G$$.

The framework is adversarial in the sense that the training procedure for $$G$$ tries to **maximize the probability of $$D$$ making a mistake**. The framework thus corresponds to a minimax two-player game.

### Related Generative Models

- Restricted Boltzmann Machines
- Deep Boltzmann Machines
- Deep Belief Networks
- Denoising Autoencoders
- Contractive Autoencoders
- Generative Stochastic Network

### Notations

- Easiest to implement GANs when the models are **multilayer perceptrons** for both generator and discriminator.
- $$p_g$$ is the generator's distribution over data $$x$$.
- $$p_z(z)$$ is an input noise function and $$G(z; \theta_g)$$ is the mapping to data space.
- $$G$$ is differentiable function represented by a paramter $$\theta_g$$.
- $$D$$ is another differentiable function that outputs a scalar.
- $$D(x)$$ represents the probability of assigning the correct label to both training examples and samples from $$G$$.
- $$G$$ is simultaneously trained to minimize $$log(1-D(G(z)))$$

### Optimization Objective

The training framework between $$D$$ and $$G$$ can be represented by a two player minimax game in value function $$V(G,D)$$,

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x\sim p_{data}(x)} [log D(x)] + \mathbb{E}_{z\sim p_z(z)} [log(1 - D(G(z)))]
$$

![GAN Architecture](/assets/2019-10-29-generative-adversarial-networks/fig-2-Generative-Adversarial-Network-GAN.png?raw=true)

### Implementation Details

- $$G$$ and $$D$$ are trained iteratively one after the other
- $$D$$ is not optimized to completion as it would lead to overfitting
- Alternate between $$k$$ steps of optimizing $$D$$ and one step of $$G$$
- Results in $$D$$ near its optimal, so long as $$G$$ changes slowly.
- Early in learning when $$G$$ is poor $$D$$ can reject samples with high confidence which causes $$log(1-D(G(z)))$$ to saturate
- Instead of minimizing $$log(1-D(G(z)))$$, maximize $$log(D(G(z)))$$ for stronger gradients early in the learning.

![Algorithm for GAN training](/assets/2019-10-29-generative-adversarial-networks/fig-1-gan-algorithm.png?raw=true)


### Theoretical Results

For a fixed $$G$$, the optimal discriminator can be found by differentiating the objective function w.r.t. $$D(x)$$. The objective function is of the form, 

$$f(y) = a\,log\,y + b\,log\,(1-y)$$

Differentiating w.r.t $y$ gives,

$$\frac{df(y)}{dy} = \frac{a}{y} - \frac{b}{1-y}$$

Since we are maximising this, the maximum can be found by estimating the point of 0 derivative, i.e,

$$
\begin{align}
    \frac{df(y)}{dy} &= 0 \\
    \frac{a}{y} - \frac{b}{1-y} &= 0 \\
    \frac{a}{y} &= \frac{b}{1-y} \\
    y &= \frac{a}{a+b}
\end{align}
$$

So the optimal discriminator for a fixed $$G$$ is given by,

$$D_G^*(x) = \frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)}$$

For this maximized $$D$$, the optimization objective can be rewritten as,

$$
C(G) = \mathbb{E}_{x\sim p_{data}} \left[log \frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)}\right] + \mathbb{E}_{x\sim p_g} \left[log\frac{p_{g }(x)}{p_{data}(x)+p_{g}(x)}\right]
$$

We can show that this expression is minimized for $$p_g=p_{data}$$. The value of $$D_G^*(x)$$ is $$1/2$$ at $$p_g=p_{data}$$ and $$C(G) = -log\,4$$.

To see that this is the minimu possible value, consider the following modification to the $$C(G)$$ expression above,

$$
\begin{align}
C(G) &= \mathbb{E}_{x\sim p_{data}} \left[log \frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)}\right] + \mathbb{E}_{x\sim p_g} \left[log\frac{p_{g }(x)}{p_{data}(x)+p_{g}(x)}\right] + log\,2 \cdot 2 - log\,4 \\
&= - log\,4 + \mathbb{E}_{x\sim p_{data}} \left[log \frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)}\right] + log\,2 +  \mathbb{E}_{x\sim p_g} \left[log\frac{p_{g }(x)}{p_{data}(x)+p_{g}(x)}\right] + log\,2 \\ 
&=- log\,4 + \mathbb{E}_{x\sim p_{data}} \left[log \frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)} + log\,2\right] +  \mathbb{E}_{x\sim p_g} \left[log\frac{p_{g }(x)}{p_{data}(x)+p_{g}(x)} + log\,2\right] \\
&=- log\,4 + \mathbb{E}_{x\sim p_{data}} \left[log \frac{p_{data}(x)}{\frac{p_{data}(x)+p_{g}(x)}{2}}\right] +  \mathbb{E}_{x\sim p_g} \left[log\frac{p_{g }(x)}{\frac{p_{data}(x)+p_{g}(x)}{2}}\right] \\
&= -log\,4 + KL\left(p_{data}||\frac{p_{data}+p_g}{2}\right) + KL\left(p_{g}||\frac{p_{data}+p_g}{2}\right)\\
&= -log\,4 + 2\cdot JSD(p_{data}||p_g)
\end{align}
$$ 


The last term is the Jensen-Shannon divergence between two distributions which is always non-negative and zero only when the two distributions are equal. So $$C^* = -log\,4$$ is the global minimum of $$C(G)$$ at $$p_g=p_{data}$$, i.e. generative model perfectly replicating the data distribution.

### Complexity Comparison of Generative Models

![](/assets/2019-10-29-generative-adversarial-networks/fig-3-comparison-of-generative models.png?raw=true)

### Disadvantages

- There is no explicit representation of $$p_g(x)$$
- $$G$$ must be synchronized well with $$D$$ during training. There are possibilities of $$D$$ being too strong leading to zero gradient for $$G$$ or $$D$$ being too weak which causes $$G$$ to collapse to many values of $$z$$ to the same value of $$x$$ which would not have enough diversity to model $$p_{data}$$

### Follow-up Citations

- RBMs and DBMs
    - A fast learning algorithm for deep belief nets by Hinton et al.
    - Deep boltzman machines by Salakhutdinov et al.
    - Information processing in dynamical systems: Foundations of harmony theory by Smolensky
- MCMC
    - Better mixing via deep representations by Bengio et al.
    - Deep generative stochastic networks trainable by backprop by Bengio et al.
- Encodings
    - What is the best multi-stage architecture for object recognition? by Jarett et al.
    - Generalized denoising auto-encoders as generative models by Bengio et al.
    - Deep sparse rectifier neural networks by Glorot et al.
    - Maxout networks by Goodfellow et al.
- Optimizations
    - Auto-encoding variational bayes by Kingma et al.
    - Stochastic backpropagation and approximate inference in deep generative models by Rezende et al.
- Learning deep architectures for AI by Bengio Y.

## REFERENCES:

<small>[Generative Adversarial Nets by Goodfellow et al.](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf){:target="_blank"}</small><br>
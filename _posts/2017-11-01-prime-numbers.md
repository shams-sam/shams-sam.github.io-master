---
layout: post
title: "The Mysterious Primes"
categories: [what-is-mathematics]
tags: [mathematics, number-system, what-is-mathematics]
description: Mathematicians have tried in vain to this day to discover some order in the sequence of prime numbers, a mystery into which the human mind will possibly never penetrate.
cover: "/assets/images/primes.png"
cover_source: "https://cdn.kastatic.org/KA-youtube-converted/9YRw0Yk7N8c.mp4/9YRw0Yk7N8c.png"
comments: true
mathjax: true
---

{% include collection-what-is-mathematics.md %}

### Introduction

In mathematics, most statements in number theory are concerned not with a single object, but with a whole class of objects that have a common property, such as the class of all even integers etc.

> Mathematics is the queen of sciences and the theory of numbers is the queen of the mathematics.

One such class of number, called the **prime numbers (or primes)** are of fundamental importance.

### The Prime Numbers

Most numbers can be resolved into smaller factors, but the ones that cannot be resolved are known as prime numbers or primes.

> A prime is an integer \\(p\\), greater than one, which has no factors other than itself and one.

A integer \\(a\\) is a factor or divisor of integer \\(b\\), if there exists an integer \\(c\\) such that \\(b=ac\\). The numbers \\(2, 3, 5, 7, \cdots \\) are primes.

So effectively, every integer can be expressed as a product of primes. And a number which not prime (other than 0 and 1) is called **composite**.

> The set of all primes is a infinite set.

The infinitude of primes is proved by contradiction. Say there exists only a finite number of primes, \\(n\\) and represented by set \\(\{p_1, p_2, \cdots p_n\}\\). Then according to the assumption any number greater than numbers in the finite set must be composite. But it is possible to come up with a number, \\(A\\) greater than all these \\(n\\) primes given by, 

$$A = p_1p_2\cdots p_n + 1 \tag{1}$$

This contracdicts the assumption of finite set of primes and hence its proved that there is a infinite set of primes.

> Every integer N greater than 1 can be factored into a product of primes in only one way.

The proof of this can be achieved by contradiction once again. If there exists a positive prime integer capable of decomposition into two essentially different products of primes, there will be a smallest such integer (using the principle of smallest integer), given by,

$$m = p_1p_2 \cdots p_r = q_1q_2 \cdots q_s \tag{2} \label{2}$$

where \\(p's\\) and \\(q's\\) are primes. By rearranging if necessary, its possible to get the following order, 

$$p_1 \leq p_2 \leq \cdots \leq p_r \text{, and } q_1 \leq q_2 \leq \cdots \leq q_s \tag{3} \label{3}$$

Where, \\(p_1 \ne q_1\\) because then one could cancel them in \eqref{2} and come up with another number smaller than \\(m\\) with two distinct prime factorization which would contradict the priniciple of smallest integer.

So, either \\(p_1 \lt q_1\\) or \\(q_1 \lt p_1\\).

Let, \\(p_1 \lt q_1\\) (if \\(q_1 \lt p_1\\), interchange the sequences in the analysis presented). Let \\(m'\\) be an integer given by,

$$m' = m - (p_1q_2\cdots q_s) \tag{4} \label{4}$$

Substituting, for \\(m\\) in \eqref{4} using \eqref{2},

$$m' = (p_1p_2 \cdots p_r) - (p_1q_2\cdots q_s) = p_1(p_2 \cdots p_r - q_2\cdots q_s) \tag{5} \label{5}$$

$$m' = (q_1q_2 \cdots q_s) - (p_1q_2\cdots q_s) = (q_1 - p_1)(q_2 \cdots q_s) \tag{6} \label{6}$$

Since \\(q_1 \gt p_1\\), it follows from \eqref{6} that \\(m'\\) is a positive integer, while from \eqref{4} it follows that \\(m' \lt m\\). Hence the prime factorization of \\(m'\\) must be unique.

From \eqref{5} \\(p_1\\) is a factor of \\(m'\\), so in \eqref{6}, \\(p_1\\) must be a factor of either \\((q_1 - p_1)\\) or \\(q_2 \cdots q_s\\). Since \\(p_1 \lt q_1\\) and \eqref{3}, \\(p_1\\) cannot be a factor of any of the \\(q's\\) nor equal to them. So, \\(p_1\\) must be a factor of \\(q_1 - p_1\\). So there exists an integer \\(h\\), such that, 

$$q_1 - p_1 = p_1 \cdot h \text{ or } q_1 = p_1(h+1) \tag{7} \label{7}$$

But, \eqref{7} is a contradicts the fact that \\(q_1\\) is a prime number. This points to the fact that the initial assumption of a number having two distinct prime factorizations must be wrong.

> If a prime \\(p\\), is a factor of the product \\(ab\\), then \\(p\\) must be a factor of either \\(a\\) or \\(b\\). 

Also, if prime factorization of \\(a\\) can be expressed as, 

$$a = p_1^{\alpha_1} p_2^{\alpha_2} \cdots p_r^{\alpha_r} \tag{8} \label{8}$$

where \\(p's\\) are distinct primes, each raised to a certain power. The number of different divisors of \\(a\\) (including \\(a\\) and 1) is given by 

$$(\alpha_1 + 1)(\alpha_2 + 1) \cdots (\alpha_r + 1) \tag{9} \label{9}$$

and all the divisors of the number \\(a\\) are given by, 

$$b = p_1^{\beta_1} p_2^{\beta_2} \cdots p_r^{\beta_r} \tag{10} \label{10}$$

where \\(\beta's\\) are integers satisfying inequalities,

$$0 \leq \beta_1 \leq \alpha_1,\,0 \leq \beta_2 \leq \alpha_2,\, \cdots 0 \leq \beta_r \leq \alpha_r$$


### Distribution of Primes

> In mathematics, the sieve of Eratosthenes is a simple, ancient algorithm for finding all prime numbers up to any given limit.

Attempts have been made at finding simple arithmetic formulas that yield only primes, even though they may not give all of them.

* **Fermat's Conjecture** : All number of the form \\(F(n) = 2^{2^n} +1\\) are primes, but was proved incorrect by Euler who discovered that \\(2^{2^5} + = 641 \cdot 6700417\\) meaning \\(F(5)\\) is not a prime. 

> Any of the numbers \\(F(n)\\) are not prime for \\(n>4\\).

There are various other similar expressions which produce primes until a certain limit.

* \\(f(n) = n^2 - n + 41\\): For limit \\(n < 41\\).

* \\(f(n) = n^2 - 79n + 1601\\): For limit \\(n < 80\\)

> It's been a futile effort to seek equation of a simple type which produces only primes. Even less promising is the attempt to find an algebraic formula that yields all the primes.


### Primes in Arithmetic Progression

> In each arithmetic progression, \\(a,\, a+d,\, a+2d,\, \cdots a+nd,\, \cdots\\), where \\(a\\) and \\(d\\) have no common factors, there are infinitely many primes.

The proof to this theorem could not be cracked for several years until finally it was presented by Lejeune Dirichlet and even after a hundred years it's not been simplified any further and would be a series of posts in itself if presented here.

Though a generalized Euclid's proof of infinitude of primes can be used to cover some special arithmetic progression such as \\(4n+3\\) and \\(6n + 5\\).

### Proof of Infinitude of Arithmetic Progressions: \\(4n+3\\)

Any prime greater than 2 must be a odd prime and hence must be of the form \\(4n+1\\) or \\(4n+3\\) for some integer \\(n\\).

Also, the product of two numbers of the form \\(4n+1\\) is again of the same form, since

$$(4a+1)(4b+1) = 16ab + 4a + 4b + 1 = 4(4ab + a + b) + 1 \tag{11} \label{11}$$

Let there be a finite number of primes in this arithmetic progression given by, \\(p_1,\, p_2,\, \cdots p_n\\) of the form \\(4n+3\\).

Consider the number,

$$N = 4(p_1p_2 \cdots p_n) - 1 = 4(p_1 \cdots p_n - 1) + 3 \tag{12} \label{12}$$

From \eqref{12}, either N has to be a prime or it can be decomposed into a product of primes, none of which can be \\(p_1,\, p_2,\, \cdots p_n\\) because they leave a remainder \\(-1\\) on dividing \\(N\\).

Also, all of the primes must not be of the form \\(4n+1\\) because the form of \\(N\\) is \\(4n+3\\) and \eqref{11}. 

Hence, one of the factors must of the form \\(4n+3\\), which is impossible, since none of the \\(p's\\), which was assumed to be the finite set of primes of the form \\(4n+3\\), can be a factor of \\(N\\) (using \eqref{12}).

So, the initial assumption that number of primes in arithmetic progression of the form \\(4n+3\\) is finite is incorrect because of the contradiction encountered.

### Proof of Infinitude of Arithmetic Progressions: \\(6n+5\\)

Any prime greater than 2 must be a odd prime and hence must be of the form \\(6n+1\\) or \\(6n+3\\) or \\(6n+5\\) for some integer \\(n\\).

Let there be a finite number of primes in this arithmetic progression given by, \\(p_1,\, p_2,\, \cdots p_n\\) of the form \\(6n+5\\).

$$(6a+1)(6b+1) = 36ab + 6a + 6b + 1 = 6(6ab + a + b) + 1 \tag{13} \label{13}$$

$$(6a+3)(6b+3) = 36ab + 18a + 18b + 9 = 6(6ab + 3a + 3b + 1) + 3 \tag{14} \label{14}$$

$$(6a+3)(6b+1) = 36ab + 6a + 18b + 3 = 6(6ab + a + 3b) + 3 \tag{15} \label{15}$$

From \eqref{13}, \eqref{14} and \eqref{15}, a number of the form \\(6n+5\\) cannot be achieved using only primes of the form \\(6n+1\\) and \\(6n+3\\).

Consider the number,

$$N = 6(p_1p_2 \cdots p_n) - 1 = 6(p_1 \cdots p_n - 1) + 5 \tag{16} \label{16}$$

From \eqref{16}, either N has to be a prime or it can be decomposed into a product of primes, none of which can be \\(p_1,\, p_2,\, \cdots p_n\\) because they leave a remainder \\(-1\\) on dividing \\(N\\).

Also, all of the primes must not be only of the form \\(6n+1\\) and \\(6n+3\\) because the form of \\(N\\) is \\(6n+5\\) and \eqref{13}, \eqref{14} and \eqref{15} show it cannot be achieved otherwise. 

Hence, one of the factors must of the form \\(6n+5\\), which is impossible, since none of the \\(p's\\), which was assumed to be the finite set of primes of the form \\(6n+5\\), can be a factor of \\(N\\) (using \eqref{16}).

So, the initial assumption that number of primes in arithmetic progression of the form \\(6n+5\\) is finite is incorrect because of the contradiction encountered.

### The Prime Number Theorem

Let \\(A_n\\) denote the number of primes among the integers \\(1, 2, 3, \cdots , n\\). 

> The density of primes among first n integers is given by the ratio \\({A_n \over n} \\).

Prime number theorem describes the asymptotic distribution of the primes among positive integers. It states that,

$$\lim_{n \to \infty} {A_n/n \over 1 / log\,n} = 1 \tag{17} $$

### Unsolved Problems Concerning Primes

* **Goldbach Conjecture:** Goldbach's conjecture is one of the oldest and best-known unsolved problems in number theory and all of mathematics. It states: Every even integer greater than 2 can be expressed as the sum of two primes.

* **Polignac's conjecture**: Twin prime conjecture, also known as Polignac's conjecture, in number theory, assertion that there are infinitely many twin primes, or pairs of primes that differ by 2.

These are simple to experiment with, but mathematically some of the most mysterious problems. These are the properties of the number system that shows that a lot is still left to be discovered about the number system.

## REFERENCES:

<small>[What is Mathematics? Second Edition - Chapter I: Natural Numbers](https://drive.google.com/open?id=0BxedRvE84NXkSy1sdzJKNDlHZGM){:target="_blank"}</small>
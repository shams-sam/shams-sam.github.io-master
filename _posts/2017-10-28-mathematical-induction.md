---
layout: post
title: Mathematical Induction
categories: [what-is-mathematics]
tags: [mathematics, number-system, what-is-mathematics, theorems]
description: Mathematical induction is a mathematical proof technique used to prove a given statement about any well-ordered set.
cover: "/assets/images/domino.jpg"
cover_source: "http://fungyung.com/data/out/34/62844389-domino-wallpapers.jpg"
comments: true
mathjax: true
---

{% include collection-what-is-mathematics.md %}

### Why Mathematical Induction?

> There are infinitely many integers.

The step-by-step procedure of passing from \\(n\\) to \\(n+1\\) which generates the infinite sequence of integers forms the basis of one of the most fundamental patterns of mathematical reasoning, the principle of mathematical induction.

Mathematical Induction is used to establish the truth of a mathematical theorem for an infinite sequence of cases.

There are two steps in proving a theorem, \\(A\\) by mathematical induction:

* The first statement \\(A_1\\) must be true.
* If a statement \\(A_r\\) is true then the statement \\(A_{r+1}\\) should be true too.

That these two conditions are sufficient to establish the truth of all the statements, \\(A_1, A_2, \cdots \\) is a logical principle which is as fundamental to mathematics as are the classical rules of Aristotelian logic.

> Suppose that a) by some mathematical argument it is shown that if \\(r\\) is any integer and if assertion \\(A_r\\) is known to be true then the truth of assertion \\(A_{r+1}\\) will follow, and that b) the first proposition \\(A_1\\) is known to be true. Then all the propositions of the sequence must be true, and A is proved.

The principle of mathematical induction rests on the fact that after any integer \\(r\\) there is a next, \\(r+1\\), and that any desired integer \\(n\\) may be reached by a finite number of such steps, starting from the integer 1.

> Although the principle of mathematical induction suffices to prove a theorem or formula once it is expressed, the proof gives no indication of how this formula was arrived at in the first place. So, it should be more fittingly called a verification.

### Proof for Arithmetic Progression

> For every value of n, the sum \\(1 + 2 + \cdots + n\\) of the first n integers is equal to \\(\frac {n(n+1)} {2} \\)

For any \\(r\\) is given by assertion \\(A_r\\) is given by,

$$1 + 2 + 3 + \cdots + r = \frac {r(r+1)} {2} \label{1} \tag{1}$$

Adding \\((r+1)\\) to both LHS and RHS of \eqref{1}, 

$$
\begin{align}
1 + 2 + 3 + \cdots + r + (r+1) &= \frac {r(r+1)} {2} + (r+1) \\
                               &= \frac {r(r+1) + 2(r+1)} {2} \\
                               &= \frac {(r+1)((r+1) + 1)} {2}
                               \label{2} \tag{2}
\end{align}
$$

\eqref{2} is nothing but assertion, \\(A_{r+1}\\).

Also for \\(r=1\\), assertion \\(A_1\\) is true because, from \eqref{1},

$$ 1 = \frac {1(1+1)} {2} \tag{3} \label{3}$$

So, using \eqref{2} and \eqref{3}, mathematical induction proves that the assertion in \eqref{1} holds for all positive integers, \\(n\\).

### Proof for Geometric Progression

The theorem states that, for every value of \\(n\\),

$$G_n = a + aq + aq^2 + \cdots + aq^n = a \frac{1 - q^{n+1}} {1-q} \label{4} \tag{4}$$

The assertion holds true for \\(n=1\\), because,

$$
\begin{align}
G_1 = a + aq &= a \frac{1 - q^{2}} {1-q} \\
             &= a \frac{(1-q) (1+q)} {1-q} \\
             &= a (1+q)  \\
    \tag{5} \label{5}
\end{align}
$$

Let's assume \\(G_r\\) is true, i.e.

$$G_r = a + aq + aq^2 + \cdots + aq^r = a \frac{1 - q^{r+1}} {1-q} \label{6} \tag{6}$$

Then, 

$$
\begin{align}
G_{r+1} &= (a + aq + aq^2 + \cdots + aq^r) + aq^{r+1} \\
        &= a \frac{1 - q^{r+1}} {1-q} + aq^{r+1} \\
        &= \frac{a - aq^{r+1} + aq^{r+1} - aq^{r+2}} {1-q} \\
        &= a \frac{1 - q^{(r+1) + 1}} {1-q} \\
        \label{7} \tag{7}
\end{align}
$$

But \eqref{7} is precisely the assertion \eqref{4} for the case \\(n=r+1\\). Hence, using \eqref{5} and \eqref{7} the assertion \eqref{4} is proved by mathematical induction.

### Proof for Sum of First n Squares

The theorem states that, for every value of \\(n\\),

$$A_n = 1^2 + 2^2 + \cdots + n^2 = \frac {n(n+1)(2n+1)} {6} \tag{8} \label{8}$$

The assertion holds true for \\(n=1\\), because,

$$
A_1 = 1^2 = \frac{1(1+1)(2(1)+1)} {6} = 1 \tag{9} \label{9}
$$

Let's assume \\(A_r\\) is true, i.e.,

$$A_r = 1^2 + 2^2 + \cdots + r^2 = \frac {r(r+1)(2r+1)} {6} \tag{10} \label{10}$$

Then,

$$
\begin{align}
A_{r+1} = (1^2 + 2^2 + \cdots + r^2) + (r+1)^2 &= \frac {r(r+1)(2r+1)} {6} + (r+1)^2 \\
&= (r+1)\frac{r(2r+1) + 6(r+1)} {6} \\
&= (r+1) \frac {2r^2 + 7r + 6} {6} \\
&= \frac{(r+1) (r+2) (2r+3)} {6} \\
\label{11} \tag{11}
\end{align}
$$

which is precisely the assertion \eqref{8} for \\(n = r+1\\). So, using \eqref{9} and \eqref{11}, mathematical induction proves the assertion.

### Proof for Sum of First n Cubes

The theorem states that, for every value of \\(n\\),

$$A_n = 1^3 + 2^3 + \cdots + n^3 = \left[\frac {n(n+1)} {2}\right]^2 \tag{12} \label{12}$$

The assertion holds true for \\(n=1\\), because,

$$
A_1 = 1^3 = \left[\frac {1(1+1)} {2}\right]^2 = 1^2 \tag{13} \label{13}
$$

Let's assume \\(A_r\\) is true, i.e.,

$$A_r = 1^3 + 2^3 + \cdots + r^3 = \left[\frac {r(r+1)} {2}\right]^2 \tag{14} \label{14}$$

Then,

$$
\begin{align}
A_{r+1} = (1^3 + 2^3 + \cdots + r^3 ) + (r+1)^3 &= \left[\frac {r(r+1)} {2}\right]^2 + (r+1)^3 \\
&= (r+1)^2\frac{r^2 + 4r + 4} {4} \\
&= (r+1)^2 \frac {(r+2)^2} {2^2} \\
&= \left[\frac {(r+1)(r+2)} {2}\right]^2
\label{15} \tag{15}
\end{align}
$$

which is precisely the assertion \eqref{12} for \\(n = r+1\\). So, using \eqref{13} and \eqref{15}, mathematical induction proves the assertion.

### Proof for Bernoulli's Inequality

The assertion, \\(A_n \\) states that, for every value of \\(n\\),

$$(1+p)^n \geq 1+np \text{, where } p>-1\tag{16} \label{16}$$

The assertion holds true for \\(n=1\\), because,

$$(1+p)^1 \geq 1+p \tag{17} \label{17}$$

Let's assume \\(A_r\\) is true, i.e.,

$$(1+p)^r \geq 1+rp \tag{18} \label{18}$$

Then, 

$$ 
\begin{align}
(1+p)^{r+1} = (1+p)^r \cdot (1+p) &\geq (1+rp)(1+p) \\
&\geq 1 + rp + p + rp^2 \\
&\geq 1+(r+1)p \text{, because } rp^2 \gt 0 \\
\label{19} \tag{19}
\end{align}
$$

which is precisely the assertion \eqref{16} for \\(n = r+1\\). So, using \eqref{17} and \eqref{19}, mathematical induction proves the assertion.

If \\(p \lt -1\\), then \\((1+p)\\) is negative and the inequality in \eqref{19} would be reversed. So, the restriction introduced in \eqref{16} is essential.

### Proof of Binomial Theorem

The assertion, \\(C_i^n \\) states that, for every value of \\(n\\),

$$C_i^n = \frac{n(n-1) \cdots (n-i+1) } {1 \cdot 2 \cdots i} = \frac {n!} {i!(n-i)!} \text{, for } i \in \{0, 1, \cdots, n\} \tag{20} \label{20}$$

The assertion holds true for \\(n=1\\), because,

$$C_i^1 = \frac{1!} {i!(1-i)!} = 1 \text{, for } i \in \{0, 1\} \tag{21} \label{21}$$

which is exactly the value for \\(C_0^1 = C_1^1 = 1\\) in Pascal's Triangle.

Let's assume \\(C_i^r\\) is true, i.e.,

$$C_i^r = \frac {r!} {i!(r-i)!} \text{, for } i \in \{0, 1, \cdots, r\} \tag{22} \label{22}$$

Then using the relation, \\(C_i^{r+1} = C_{i-1}^{r} + C_i^{r} \text{, } C_i^{r+1}\\) is given by,

$$
\begin{align}
C_i^{r+1} &= C_{i-1}^{r} + C_i^{r} \\
&= \frac{r!}{(i-1)!(r - i + 1)!} + \frac{r!}{i!(r-i)!} \\
&= \frac{r!}{(i-1)!(r-i)!} \left[ \frac{1}{r-i+1} + \frac{1}{i} \right] \\
&= \frac{r!}{(i-1)!(r-i)!} \left[ \frac{r+1}{i(r-i+1)} \right] \\
&= \frac{(r+1)!}{(i)!((r+1)-i)!} \\
\tag{23} \label{23}
\end{align}
$$

which is precisely the assertion \eqref{20} for \\(n = r+1\\). So, using \eqref{21} and \eqref{23}, mathematical induction proves the assertion.

### Generalized Mathematical Induction

> If a sequence of statements \\(A_s, A_{s+1}, \cdots\\) is given where \\(s\\) is a some positive integer, and if <br>
> a) For every value \\(r \geq s\\), the truth of \\(A_{r+1}\\) will follow from the truth of \\(A_r\\),<br>
> b) \\(A_s\\) is known to be true, <br>
> then all the statements \\(A_s, A_{s+1}, \cdots\\) are true; i.e. \\(A_n\\) is true for all \\(n \geq s\\)

### Proof for Bernoulli's Inequality (Strict version)

The assertion, \\(A_n \\) states that, for every value of \\(n\\),

$$(1+p)^n \gt 1+np \text{, for } p>-1 \text{ and } p\ne0 \text{ and } n \geq 2\tag{24} \label{24}$$

The assertion holds true for \\(n=s=2\\), because,

$$(1+p)^2 = 1 + 2p + p^2 \gt 1+2p \text{, because } p \ne 0 \text{, so } p^2 \gt 0   \tag{25} \label{25}$$

Let's assume \\(A_r\\) is true, i.e.,

$$(1+p)^r \gt 1+rp \tag{26} \label{26}$$

Then, 

$$ 
\begin{align}
(1+p)^{r+1} = (1+p)^r \cdot (1+p) &\gt (1+rp)(1+p) \\
&\gt 1 + rp + p + rp^2 \\
&\gt 1+(r+1)p \text{, because } rp^2 \gt 0 \\
\label{27} \tag{27}
\end{align}
$$

which is precisely the assertion \eqref{24} for \\(n = r+1\\). So, using \eqref{25} and \eqref{27}, mathematical induction proves the assertion.

If \\(p \lt -1\\), then \\((1+p)\\) is negative and the inequality in \eqref{19} would be reversed. So, the restriction introduced in \eqref{16} is essential.


### Principle of Smallest Integer

> Every non-empty set \\(C\\) of positive integers has a smallest number. (Set may be finite or infinite.)

At first it seems like a trivial principle but it actually does not apply to many sets that are not integers, e.g. the set of positive fractions \\(1, {1 \over 2} {1 \over 3} \cdots \\) does not contain a smallest number.

### Proof for Mathematical Induction

The principle of smallest integer can be used to prove the principle of mathematical induction as a theorem.

Let us consider any sequence of statements \\(A_1, A_2, \cdots \\) such that, 

1. For any integer \\(r\\) the truth of \\(A_{r+1}\\) will follow from that of \\(A_r\\)
2. \\(A_1\\) is known to be true.

if 1 of the \\(A's\\) were false, the set \\(C\\) of all positive integers \\(n\\) for which \\(A_n\\) is false would be non-empty. By the principle of smallest integer, \\(C\\) would have a smallest integer \\(p\\), which must be greater than 1 because of condition (2) in the theorem. Hence, \\(A_p\\) would be false, but \\(A_{p-1}\\) true, which contradicts condition (1) of the theorem. So, the assumption in the first place, that any one of the \\(A's\\) is false is untenable.

> Mathematical Induction must be applied very carefully to prove an assertion because, it is often fallacious because of a misleading base case. A popular example of this is "All number are equal fallacy".

## REFERENCES:

<small>[What is Mathematics? Second Edition - Chapter I: Natural Numbers](https://drive.google.com/open?id=0BxedRvE84NXkSy1sdzJKNDlHZGM){:target="_blank"}</small>
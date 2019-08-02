---
layout: post
title: "Congruence and Modular Arithmetic"
categories: [what-is-mathematics]
tags: [mathematics, number-system, what-is-mathematics]
description: If two numbers have the property that their difference is integrally divisible by a number (i.e., is an integer), then they are said to be "congruent modulo". 
cover: "/assets/images/clock.jpg"
cover_source: "http://wallpaperswide.com/old_clock-wallpapers.html"
comments: true
mathjax: true
---

{% include collection-what-is-mathematics.md %}

### Congruence

Two integers \\(a\\) and \\(b\\) are **congruent modulo \\(d\\)**, where \\(d\\) is a fixed integer, if \\(a\\) and \\(b\\) leave same remainder on division by \\(d\\), i.e.

$$a-b = nd \tag{1} \label{1}$$

It is denoted by,

$$ a\equiv b\pmod d \tag{2} \label{2}$$

Following defination of congruences are equivalent:

* \\(a\\) is congruent to \\(b\\) modulo \\(d\\).
* \\(a = b + nd\\) for some integer n.
* \\(d\\) divides \\(a-b\\).

### Properties and Proof

Congruence with respect to a fixed modulus has many of the formal properties of ordinary equality.

* \\(a\equiv a\pmod d\\)
* If \\(a\equiv b\pmod d\\), then \\(b\equiv a\pmod d\\)
* If \\(a\equiv b\pmod d\\) and \\(b\equiv c\pmod d\\), then \\(a\equiv c\pmod d\\)

If \\(a\equiv a'\pmod d\\) and \\(b\equiv b'\pmod d\\), then

$$a + b\equiv a' + b'\pmod d \tag{3} \label{3}$$

$$a - b\equiv a' - b'\pmod d \tag{4} \label{4}$$

$$ab\equiv a'b'\pmod d \tag{5} \label{5}$$

Say, 

$$ a = a' + rd $$

$$ b = b' + sd $$

then,

$$ a + b = a' + b' + (r+s)d$$

$$ a - b = a' - b' + (r-s)d$$

$$ ab = a'b' + (a's + b'r +rsd)d$$

### Geometric Interpretation

Generally, integers are represented geometrically using a number line, where a segment of unit length is chosen and multiplied in either directions to represent negative or positive integers.

![Geometric Representation of Congruence](/assets/2017-12-20-congruence-and-modulo/fig-1-geometric-representation.svg?raw=true)

But, when an integer modulo \\(d\\) is considered, the magnitude is insignificant as long as the behavior on division by \\(d\\) is same (i.e. they leave the same remainder on division by \\(d\\)). This is geometrically represented using a circle divided into d equal parts. This is because any integer divided by \\(d\\) leaves as remainder one of the \\(d\\) numbers \\(0, 1, \cdots, d-1\\) which are placed at equal distances on the circumference of the circle. Every integer is congruent modulo \\(d\\) to one of these numbers and hence can be represented by one of these points. (**Two numbers are congruent if they occur at the same point the circle.**)

### Application of Congruence Properties

> The test for divisibility, generally taught in elementary school, is a direct result of the properties of congruence operation.

For example, 

$$10 \equiv -1 \pmod{11}$$

since \\(10 = -1 + 11\\). Successively multiplying this congruence, using \eqref{5}, we obtain, 

$$10^2 \equiv (-1)(-1) = 1 \pmod{11}$$

$$10^3 \equiv (-1)(-1)(-1)= -1 \pmod{11}$$

$$10^4 \equiv (-1)(-1)(-1)(-1)= 1 \pmod{11}$$

$$\vdots$$

So using \eqref{3} and \eqref{5}, it can be shown that any two number, z and t of the form shown below will leave the same remainder when divided by 11.

$$z = a_0 + a_1\cdot 10 + a_2 \cdot 10^2 + \cdots + a_n \cdot 10^n \tag{6} \label{6} $$

$$t = a_0 - a_1 + a_2 - \cdots + (-1)^n \cdot a_n \tag{7} \label{7} $$

Here, \\(z\\) is the format of any integer to the base 10. Hence, a number is divisible by 11 (i.e. leaves a remainder 0), if and only if \\(t\\) is divisible by 11 (which in \eqref{7} basically means that **the difference of the sum of all the odd digits and even digits together should be divisible by 11**, including 0.)

It can be observed that while such patterns  are easier for numbers like 3, 9, 11, they are not easy to remember for other numbers like 7 and 11, as shown below.

$$1 \equiv 1 \pmod{13}$$ 

$$10 \equiv -3 \pmod{13}$$ 

$$10^2 \equiv -4 \pmod{13}$$

$$10^3 \equiv -1 \pmod{13}$$

$$10^4 \equiv 3 \pmod{13}$$

$$10^5 \equiv 4 \pmod{13}$$

$$10^6 \equiv 1 \pmod{13}$$

$$t = a_0 - 3 \cdot a_1 - 4 \cdot a_2 - a_3 + 3 \cdot a_4 + 4 \cdot a_5 + a_6 - \cdots \tag{8} \label{8} $$

From above we reach the result that any number \\(z\\) in \eqref{6} is divisible by 13 if and only if \\(t\\) of the form \eqref{8} is divisible by 13 (**clearly not an easy one to remember :P**).

Using a similar approach one can deduce the divisibility rule for any other integer.

### Other Properties

* $$ab\equiv 0 \pmod d \tag{9} \label{9}$$ 

only if either \\(a\equiv 0 \pmod d\\) or \\(b\equiv 0 \pmod d\\). Property only holds if \\(d\\) is a prime number. If \\(d\\) was a composite, there exist numbers \\(a \lt d\\) and \\(b \lt d\\), such that,

$$d = a\cdot b$$

Where,
$$\require{cancel}$$

$$a \cancel{\equiv} 0 \pmod d \text{ and } b \cancel{\equiv} 0 \pmod d$$

But,

$$ a \cdot b = d \equiv 0 \pmod d$$

* **Law of Cancellation**: With respect to a prime modulus, if \\(ab \equiv ac\\) and \\(a \cancel{\equiv} 0\\), then \\(b \equiv c\\).

### Fermat's Theorem

If \\(p\\) is any prime which does not divide the integer \\(a\\), then 

$$a^{p-1} \equiv 1 \pmod p \tag{10} \label{10}$$

Consider multiples of \\(a\\),

$$m_1 = a, m_2 = 2a, m_3 = 3a, \cdots m_{p-1} = (p-1)a \tag{11} \label{11}$$

Let two of these numbers, \\(m_r\\) and \\(m_s\\) be congruent modulo \\(p\\), then,

\\(p\\) must be a factor of \\(m_r - m_s = (r-s)a\\) for some \\(r, s\\) such that \\(1 \leq r \lt s \leq (p-1)\\).

But since it is assumed that \\(p\\) does not divide \\(a\\) and also \\(p\\) cannot be factor of \\(r-s\\) since it is less than \\(p\\).

From \eqref{9}, it can be concluded that two numbers from \eqref{11} cannot be congruent modulo \\(p\\).

So each of the numbers in \eqref{11} must be congruent to \\(1, 2, 3, \cdots , (p-1)\\) in some arrangement. So,

$$m_1 m_2 \cdots m_{p-1} = 1 \cdot 2 \cdots (p-1) a^{p-1} \equiv 1 \cdot 2 \cdots (p-1) \pmod p \tag{12} \label{12}$$

For simplicity, let \\(K = 1 \cdot 2 \cdots (p-1)\\), then

$$K(a^{p-1}-1) \equiv 0 \pmod p \tag{13} \label{13}$$

where \\(K\\) is not divisible by \\(p\\), since none of its factors are, hence from \eqref{9}, \\((a^{p-1} - 1)\\) must be divisible by \\(p\\), i.e.

$$a^{p-1} -1 \equiv 0 \pmod p \tag{14} \label{14}$$

Hence, proving \eqref{10}.

## REFERENCES:

<small>[What is Mathematics? Second Edition - Chapter I: Natural Numbers](https://drive.google.com/open?id=0BxedRvE84NXkSy1sdzJKNDlHZGM){:target="_blank"}</small>

---
layout: post
title: "Congruence and Modular Arithmetic"
categories: []
tags: [mathematics, number system, what is mathematics]
description: If two numbers and have the property that their difference is integrally divisible by a number (i.e., is an integer), then and are said to be "congruent modulo". 
cover: "/assets/images/clock.jpg"
cover_source: "http://wallpaperswide.com/old_clock-wallpapers.html"
comments: true
mathjax: true
---

{% include what-is-mathematics.md %}

### Congruence

Two integers \\(a\\) and \\(b\\) are **congruent modulo \\(d\\)**, where \\(d\\) is a fixed integer, if \\(a\\) and \\(b\\) leave same remainder on division by \\(d\\), i.e.

$$a-b = nd \tag{1} \label{1}$$

It is denoted by,

$$ a\equiv b\pmod d$$

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

* \\(a + b\equiv a' + b'\pmod d\\)
* \\(a - b\equiv a' - b'\pmod d\\)
* \\(ab\equiv a'b'\pmod d\\)

Say, 

$$ a = a' + rd $$

$$ b = b' + sd $$

then,

$$ a + b = a' + b' + (r+s)d $$

$$ a - b = a' - b' + (r-s)d $$

$$ ab = a'b' + (a's + b'r +rsd)d $$

### Geometric Interpretation

Generally, integers are represented geometrically using a number line, where a segment of unit length is chosen and multiplied in either directions to represent negative or positive integers.

![Geometric Representation of Congruence](/assets/2017-12-20-congruence-and-modulo/fig-1-geometric-representation.svg?raw=true)

But, when an integer modulo \\(d\\) is considered, the magnitude is insignificant as long as the behavior on division by \\(d\\) is same (i.e. they leave the same remainder on division by \\(d\\)). This is geometrically represented using a circle divided into d equal parts. This is because any integer divided by \\(d\\) leaves as remainder one of the \\(d\\) numbers \\(0, 1, \cdots, d-1\\) which are placed at equal distances on the circumference of the circle. Every integer is congruent modulo \\(d\\) to one of these numbers and hence can be represented by one of these points. (**Two numbers are congruent if they occur at the same point the circle.**)


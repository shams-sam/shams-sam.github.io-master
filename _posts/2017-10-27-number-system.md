---
layout: post
title: "What is Number System ?"
categories: [what-is-mathematics]
tags: [mathematics, number-system, what-is-mathematics]
description: A number system is a way to represent numbers. We are used to using the base-10 number system, which is also called decimal.
cover: "/assets/images/number-system.jpg"
cover_source: "http://jonvilma.com/images/binary-9.jpg"
comments: true
mathjax: true
---

{% include collection-what-is-mathematics.md %}

### The Natural Numbers

The natural numbers were created by the human mind out of the necessity to count the objects around us. Most basics of mathematics are associated with association of numbers with tangible objects. But advanced mathematics is built on top of the **abstract concept of number system**.

> God created natural numbers; everything else is man's handiwork.

### Laws of Arithmetic

Natural numbers have only **two basic operations**, namely, addition and multiplication. The mathematical theory of the natural numbers or positive integers is known as **arithmetic**. Arithmetic is based on the fact that operations on numbers are governed by certain laws.

(a, b, c ... symbolically denote integers)

The fundamental laws of arithmetic are:

* **Commutative** law, i.e. \\(a+b = b+a\\) and \\(ab = ba\\)
* **Associative** law, i.e. \\(a + (b + c) = (a + b) + c\\) and \\(a(bc) = (ab)c\\)
* **Distributive** law, i.e. \\(a(b+c) = ab + ac\\)

Addition and subtraction are **inverse operations** because, \\((a+d)-d = a\\). Similarly, multiplication and division are inverse because \\({a \over d} \cdot d = a\\). Also, **0 and 1 are the identities of operations addition and multiplication respectively**.

### Representation of Integers

A number system has a set of **digit symbols and numbers** where digit symbols are used to denote the larger numbers not available directly in the set of digit symbols. Modern number systems are associated with the place values of individual digits in the number, such as,

$$ 372 = 3 \cdot 10^2 + 7 \cdot 10^1 + 2 \cdot 10^0$$

These are called **positional notations**. Here, a large number can also be represented using the basic symbols from the set of digit symbols. It is useful to have a way of indicating the result in a general form using a uniform logic, i.e. a general method for representing an integer, \\(z\\) in the decimal system is to express it as follows

$$z = a_n \cdot 10^n + a_{n-1} \cdot 10^{n-1} + \cdots + a_1 \cdot 10 + a_0$$

which would be represented by the symbol \\(a_na_{n-1}\cdots a_1a_0\\)

Specifically, in case of decimal system the base is 10 as can be seen in the equations above. But, in general, for a number system **any number greater than 1 can serve as the base** of the system. E.g. a **septimal** system has base 7 and an integer is expressed as,

$$b_n \cdot 7^n + b_{n-1} \cdot 7^{n-1} + \cdots + b_1 \cdot 7 + b_0$$

and it would be represented by the symbol \\(b_nb_{n-1}\cdots b_1b_0\\)

As a result 109 in decimal system is represented by 214 in septimal system for the reason shown below 

$$2 \cdot 7^2 + 1 \cdot 7 + 4 = 109$$

> As a general rule, in order to convert from base 10 to base B, perform successive divisions of number z by B; the remainders in reverse order would be the number representation in the system with base B.

Too small a base base has disadvantages (even **a small number like 72 has a lengthy representation 1,001,111 in the dyadic system i.e. the binary system**), while a large base requires learning many digit symbols, and an extended multiplication table. 

> The duodecimal system, i.e., choice of base 12 is advocated in many places, because it is exactly divisible by two, three, four, and six. Due to this property, works involving division and fraction are simplified.

Early systems of numerations were not positional in nature, but were based on purely additive principle. E.g. in roman symbolism, 

$$CXVIII = \text{one hundred} + \text{ten} + \text{five} + \text{one} + \text{one} + \text{one}$$

i.e., the symbol \\(CXVIII\\) represents 118. Same was true about other early number systems like the Egyptian, Hebrew, Greek systems of numeration.

> Disadvantage of any purely additive notation is that it requires more and more number of new symbols as the number to be represented gets larger. Also, the computation with additive systems is so difficult to perform, that computation was confined to a few adepts in the ancient times.

The positional system has the property that all numbers, however large or small, can be represented by the use of a small set of different digit symbols. Also, the major advantage lies in the **ease of computation**.

### Computation in Other Number Systems

A very curious property of the decimal number system that points to usage of other number systems in the past is the number words used. E.g. the words for 11 and 12 are not constructed as the other teens are, suggesting a linguistic independence from words used for 10. Such peculiarities suggest remnants of use of other bases, notably 12 and 20.

Words vingt and quartevingt used for 20 and 80 respectively in French suggest a possibility of number with **base 20**. There are traces of Babylonian astronomers using a number system called **sexagesimal** (base 60).

> The tables for addition and multiplication change with number system while the rules of arithmetic remain the same.

Below is the **table for addition** followed by **table for multiplication** in duodecimal number system. (**10 and 11 in the duodecimal representation below are represented using A and B respectively.**)

| + | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | A | B |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | A | B | 10 |
| 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | A | B | 10 | 11 |
| 3 | 4 | 5 | 6 | 7 | 8 | 9 | A | B | 10 | 11 | 12 |
| 4 | 5 | 6 | 7 | 8 | 9 | A | B | 10 | 11 | 12 | 13 |
| 5 | 6 | 7 | 8 | 9 | A | B | 10 | 11 | 12 | 13 | 14 |
| 6 | 7 | 8 | 9 | A | B | 10 | 11 | 12 | 13 | 14 | 15 |
| 7 | 8 | 9 | A | B | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
| 8 | 9 | A | B | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
| 9 | A | B | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
| A | B | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
| B | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 1A | 

| . | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | A | B |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | A | B |
| 2 | 2 | 4 | 6 | 8 | A | 10 | 12 | 14 | 16 | 18 | 1A |
| 3 | 3 | 6 | 9 | 10 | 13 | 16 | 19 | 20 | 23 | 26 | 29 |
| 4 | 4 | 8 | 10 | 14 | 18 | 20 | 24 | 28 | 30 | 34 | 38 |
| 5 | 5 | A | 13 | 18 | 21 | 26 | 2B | 34 | 39 | 42 | 47 |
| 6 | 6 | 10 | 16 | 20 | 26 | 30 | 36 | 40 | 46 | 50 | 56 |
| 7 | 7 | 12 | 19 | 24 | 2B | 36 | 41 | 48 | 53 | 5A | 65 |
| 8 | 8 | 14 | 20 | 28 | 34 | 40 | 48 | 54 | 60 | 68 | 74 |
| 9 | 9 | 16 | 23 | 30 | 39 | 46 | 53 | 60 | 69 | 76 | 83 |
| A | A | 18 | 26 | 34 | 42 | 50 | 5A | 68 | 76 | 84 | 92 |
| B | B | 1A | 29 | 38 | 47 | 56 | 65 | 74 | 83 | 92 | 101 | 

## REFERENCES:

<small>[What is Mathematics? Second Edition - Chapter I: Natural Numbers](https://drive.google.com/open?id=0BxedRvE84NXkSy1sdzJKNDlHZGM){:target="_blank"}</small>
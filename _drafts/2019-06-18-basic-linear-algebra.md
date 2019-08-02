---
layout: post
title: "Basics of Linear Algebra"
categories: [basics-of-linear-algebra]
tags: [mathematics, linear-algebra]
description: Linear algebra is the branch of mathematics concerning linear equations, linear functions and their representations through matrices and vector spaces. Linear algebra is central to almost all areas of mathematics.
cover: "/assets/images/linear-algebra.png"
cover_source: "https://ka-perseus-images.s3.amazonaws.com/6eca6b57567643dd7743f2efc5c90e6bad3133d8.png"
comments: true
mathjax: true
---

### Active Recall
1. [What is a vector space?](#vector-spaces)
2. [What is isomorphism, bijective linear maps and kernel of a map?](#linear-maps)

### Introduction

- **Linear algebra** is the branch of mathematics concerning linear equations such as \\(a_1 x_1 + \cdots + a_n x_n = b \\), linear functions such as \\(\(x_1, \cdots, x_n\) \mapsto a_1 x_1 + \cdots + a_n x_n\\) and their representations through matrices and vector spaces. It is fundamental in almost all areas of mathematics, including modern presentation of geometry, defination of basic objects such as lines, planes and rotations. It is also user as a first order approximation of non-linear systems that can not be modeled using linear algebra.
- Historically, linear algebra was introduced through system of linear equations and matrics. In the current times, linear algebra presented through vector spaces is preferred as it is more general (not limited to finite-dimensional spaces), conceptually simpler, although more abstract.

### Vector Spaces

- **Field**: A set on which addition, subtraction, multiplication and division are defined, and behave as the corresponding operations on rational and real numbers do. E.g. field of rational number, field of real number etc.
- **Set**: A collection of distinct object that is considered as an object itself. E.g. the natural numbers such as 1, 2, 3 etc can be considered as objects, but the group of natural numbers (finite or infinite) is collectively an object called set.
- **Abelian Group**: Also called **commutative group**, it refers to group in which the results of applying the operation to two group elements does not depend on the order in which they are written, i.e. these groups obey the laws of commutativity. 
- **Vector Space**: A set \\(V\\) defined over field \\(F\\), equipped with two binary operations satisfying the few axioms. Elements of \\(V\\) are called **vectors** and elements of \\(F\\) are called **scalars**. 
- The binary operations on vector spaces are as follows:
    + **Vector Addition**: operation takes two vectors \\(v\\) and \\(w\\) and outputs a third vector \\(v+w\\)
    + **Scalar Multiplication**: operation takes any scalar \\(a\\) and any vector \\(v\\) and outputs a new vector \\(av\\)
- The axioms that the above operations must satisfy to qualify as vector spaces are as follows:
    + **Associativity** of addition: \\(u+\(v+w\) = \(u+v\)+w\\)
    + **Commutativity** of addition: \\(u+v = v+u\\)
    + **Identity Element** of addition: There exists an element \\(0\\) in \\(V\\), called **zero vector** (or zero), such that \\(v+0=v \, \forall \, v \in V\\) 
    + **Inverse Element** of addition: There exists an element \\(-v \in V\\), called **additive inverse** of \\(v\\), such that \\(v+\(-v\)=0 \, \forall \, v \in V\\)
    + **Distributivity** of scalar multiplication w.r.t. vector addition: \\(a\(u+v\)=au+av\\)
    + **Distributivity** of scalar multiplication w.r.t. field addition: \\(\(a+b\)v=av+bv\\)
    + **Compatibility** of scalar multiplication with field multiplication: \\(a\(bv\)=ab\(v\)\\)
    + **Identity Element** of scalar multiplication: \\(1v=v\\), where \\(1\\) denotes the **multiplicative identity** of \\(F\\). 
- As a result of the first 4 axioms, \\(V\\) is an abelian group under addition.

### Linear Maps

- **Morphism**: Sharing some overlap with the term **function** in mathematics, morphism refers to a structure-preserving map between objects of same type. In set theory, morphisms are function, in linear algebra they are linear transformations etc.
- **Isomorphism**: A homomorphism or morphism that can be reversed by an inverse morphism. Two mathematical objects are isomorphic if an isomorphism exists between them. An **autoisomorphism** is one where source and target coincide.
- **Function**: A relation between sets, that asssociates every element of a set (domain) to exactly one element of another set (codomain). E.g. integer to integer functions, or real number to real number functions.
- **Map**: Also termed as **mapping**, it refers to a relationship between mathematical objects or structures. Maps may either be functions or morphisms.
- **Gaussian Elimination**:  Also called row reduction, it is an algorithm in linear algebra for solving system of linear equations. It is usually a sequence of operations performed on the corresponding matrix of coefficients.

- Linear maps are mappings between vector spaces that **preserve the vector-space structure**. Given two vector spaces \\(V\\) and \\(W\\) over field \\(F\\), a linear map (also called, in some contexts, linear transformation, linear mapping and linear operator) is a map \\(T \colon V \to W\\) that is compatible with addition and scalar multiplication, i.e. \\(T\(u+v\)=T\(u\)+T\(v\)\\) and \\(T\(av\)=aT\(v\)\\) for any vector \\(u,\, v\\) in \\(V\\) and a scalar \\(a\\) in \\(F\\).
- When a **bijective** linear map exists between two vector spaces (i.e. every vector from the second space is associated with exactly one in the first), the two spaces are isomorphic.
- The **kernel** of a linear map \\(L \colon V \to W\\) between two vector spaces \\(V\\) and \\(W\\), is the set of all elements \\(v \in V\\) for which \\(L\(v\)=0\\), where \\(0\\) denotes the zero vector in \\(W\\), i.e.,

    $$ker(L) = \{v \in V | L(v)=0\}$$

## REFERENCES:

<small>[Quantifying Political Leaning from Tweets, Retweets, and Retweeters](https://ieeexplore.ieee.org/abstract/document/7454756){:target="_blank"}</small><br>
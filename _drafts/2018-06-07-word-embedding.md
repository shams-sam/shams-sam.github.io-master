---
layout: post
title: "Word Embedding"
categories: []
tags: [NLP, machine-learning, papers, word-embedding]
description: The collective name for a set of language modeling and feature learning techniques in NLP where words or phrases from the vocabulary are mapped to vectors of real numbers.
cover: "/assets/images/words.jpeg"
cover_source: "https://cdn-images-1.medium.com/max/1600/1*lR1etLrPtTcYRNj9-YefTQ.jpeg"
comments: true
mathjax: true
---

{% include collection-distributed-vector-representation.md %}

### Introduction

The need of vector representation for words arises from the fact that most of the NLP and machine learning techniques operate on fixed-length numerical values (except algorithms like decisions trees etc. which can operate on raw text as well). 

> The conversion of words into numbers or vector of number is termed as feature extraction or feature encoding.

Bag of Words (BoW) is one of the most popular feature encoding techniques in NLP that is simple to understand, easy to implement and gives promising results on a large dataset (that helps generalization over varieties). 

### Bag of Words Model

The bag of words is a representations of text that describes the occurences of words in a sentence using two attributes:

* **Vocabulary**: is the set of all known words.
* **Word Occurence Measure (Score)**: defines the words that are present in a document.

> The word "bag" symbolizes the fact that the order/structure of words represented by this model is lost in the BoW representation.

This model stems from the observation that documents with similar words have similar content.

The implementation of a BoW model can vary in a lot of ways as one can customize the way the vocabulary is built or the way occurences of words scored.

* Vocabulary can be built by taking all unique words or considering n-grams for frequently occuring phrases, with or without case sensitivity while ignoring or keeping the punctuations.

* **Scoring:**
  * Simplest way is to score the presence of word as 1 and absence as 0.
  * Apart from 1/0 encoding, one can score the presence with term count, term frequency, tf-idf etc.

As the size of vocabulary increases, the length of vector representing a document also increases. As a direct result of this one would observe that BoW model over a big corpus leads to a sparse matrix, which increase memory utilization and affect the computation cost as well.

In order to cope with this one can limit vocabulary by following basic approaches:
* Ignore case
* Ignore punctuations
* Remove stopwords
* Avoiding spell errors
* stemming or lemmatization
* incorporate n-grams

**Word hashing** can also be used to limit the vocabulary. But one has to consider the trade-off between probability of collision and sparsity. This is called the **hashing trick** or **feature hashing**.

Limitations of BoW model:
* Vocabulary selection must be done carefully as it may lead to exploding sparsity.
* Sparsity of the encoded space cannot be avoided most BoW models.
* Discarding of word order often times leads to loss of meaning and context.

### Skip Gram 



## REFERENCES:

<small>[A Gentle Introduction to the Bag-of-Words Model](https://machinelearningmastery.com/gentle-introduction-bag-words-model/){:target="_blank"}</small><br>
<small>[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf){:target="_blank"}</small>
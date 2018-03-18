---
layout: post
title: "What is Computer Vision?"
categories: [basics-of-computer-vision]
tags: [machine-learning, computer-vision, image-processing, udacity]
description: The goal of computer vision is to write computer programs that can interpret images.
cover: "/assets/images/computer-vision.jpg"
cover_source: "https://cdn.nanalyze.com/uploads/2016/04/Computer-Vision-Image-Understanding-teaser.jpg"
comments: true
mathjax: true
---

{% include collection-basics-to-computer-vision.md %}

### What is Computer Vision?

Computer vision is concerned with the automatic extraction, analysis and understanding of useful information from a single image or a sequence of images such as a video.

### What is Computational Photography?

Computational photography refers to analysis, manipulation and synthesis of images using numerical algorithms. It combines methodologies from image processing, computer vision, computer graphics and photography.

### Applications of Computer Vision

* OCR and Face Recognition
* Object Recognition
* Special Effects and 3D Modeling
* Smart Cars and Sports
* Vision based computer interactions
* Security and Medical Imaging

### Why is Computer Vision Hard?

In order to understand why computer vision is hard, one has to familiarize themselves with the difference between measurements of metrics of an image and the perceptions that we draw from them. Essentially if one looks at the image below, it would seem that the boxes A and B are of different shade (essentially box A seems darker than box B).

![Fig. 1 - Difference in Perception](/assets/2018-02-01-what-is-computer-vision/fig-1-difference-of-perception.png?raw=true)

But, in reality if we place a grayscale intesity matcher for comparison of the block shades, it is seen that the two intensities are the same as seen in the image below.

![Fig. 2 - Uniformity of Measurement](/assets/2018-02-01-what-is-computer-vision/fig-2-uniformity-of-measurements.png?raw=true)

Another classic example showing the way perception differs based on image manipulation can be seen below.

![Fig. 3 - Ball in a Box - Shadow Manipulation](/assets/2018-02-01-what-is-computer-vision/fig-3-ball-in-a-box.gif?raw=true)

The shadow manipulation demo by Kersten Labs shows an apt example of how brain changes perception on slight changes in visual input to match the accepted norms.

It is these intricate details and variations among them that make the problem of computer vision a challenging one.


## REFERENCES:

<small>[Introduction to Computer Vision - Udacity](https://classroom.udacity.com/courses/ud810){:target="_blank"}</small>
---
layout: post
title: "Basics of Computer Vision"
categories: []
tags: [machine learning, computer vision, image processing, udacity]
description: The goal of computer vision is to write computer programs that can interpret images.
cover: "/assets/images/computer-vision.jpg"
cover_source: "https://cdn.nanalyze.com/uploads/2016/04/Computer-Vision-Image-Understanding-teaser.jpg"
comments: true
mathjax: true
---

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

![Difference in Perception](/assets/2018-02-04-basics-of-computer-vision/fig-1-difference-of-perception.png?raw=true)

But, in reality if we place a grayscale intesity matcher for comparison of the block shades, it is seen that the two intensities are the same as seen in the image below.

![Uniformity of Measurement](/assets/2018-02-04-basics-of-computer-vision/fig-2-uniformity-of-measurements.png?raw=true)

Another classic example showing the way perception differs based on image manipulation can be seen below.

![Ball in a Box - Shadow Manipulation](/assets/2018-02-04-basics-of-computer-vision/fig-3-ball-in-a-box.gif?raw=true)

The shadow manipulation demo by Kersten Labs shows an apt example of how brain changes perception on slight changes in visual input to match the accepted norms.

It is these intricate details and variations among them that make the problem of computer vision a challenging one.

### Images as Function

Images are generally associated with concepts of vision and perception. But in the field of computer vision it is important to understand that image can be represented as a function as well. An image can be represented as a function \\(I(x, y)\\) which gives the intensity represented by height of the bar for a given coordinate \\(x\\) and \\(y\\).

So basically, an image can be thought of as one of the following:

* a matrix of numbers 
* pixel intensity as a function of coordinates
* output of a camera

> Theoretically, an image can be modelled as a function \\(f\\) or \\(I\\) from \\(\mathbb{R}^2 \to \mathbb{R}\\) where \\(f(x,y)\\) gives the intensity or value at position \\((x,y)\\).

Practically, an image can be modelled over a rectangle with a finite range, given by:

$$ f: [a,b] x [c,d] \to [min, max] \label{1} \tag{1} $$

Similarly, color images are three functions stacked together representing the three channels (i.e. often R, G and B), written as, vector-valued functions (i.e. every pixel is a vector of numbers), 

$$ f(x,y) = [r(x,y); g(x,y); b(x,y)] \label{2} \tag{2} $$

Such as function can be generalized as,

$$ f: \mathbb{R}^2 \to \mathbb{R}^3 \label{3} \tag{3}$$

### Digital Images

Another important realization in the computer vision is the fact that images in a computer system are discrete images and not continuous ones like human eyes see, i.e. images in computer have a matrix representation with non-continuous intensity values between two adjacent pixel which might not be the case for the actual ground truth image it is capturing.

Discretization of the digital image is two fold:

* Sample the actual image onto a 2D space of a regular matrix
* Quantize the intensity values (as digital images do not take continuous values for intensities) to the nearest integer.

Such operations often will lead to loss of some information but if often minimal and can be worked with.

### Loading and Exploring Image

```octave
% import image
img = imread('image.jpg');
imshow(img);

% load the image package
pkg load image;

% size of 3 channel image
disp(size(img));

% red channel extraction
img_red = img(:, :, 1);
imshow(img_red);
plot(img_red(80, :));

% convert to grayscale
img = rgb2gray(img)

% size and class of image
disp(size(img));
disp(class(img));

% extract intensities from image
disp(img(50, 100));
plot(img(50, :));

% cropping image by slicing
disp(gry(101:103, 201:203));
```

* Slicing a matrix is same as cropping.
* Multiplying a matrix with a scalar (i.e scaling an image) helps adjust the brightness, i.e. if scalar greater than 1, the image becomes brighter else it gets darker because pixel value 255 represents white and 0 represents black.
* Alpha blending has roots in scaled addition to two images to maintain pixel limits.

### Noise in Image

Noise in an image can be represented with a function, i.e.

$$ I'(x,y) = I(x,y) + \eta(x,y) $$

where \\(\eta\\) is the noise.

Common types of noise functions are:

* **Salt and pepper noise** is random occurences of white and black pixels.
* **Impulse noise** has random occurences of white pixels only.
* **Guassian noise** has variations in intensity drawn from a Guassian normal distribution.

## REFERENCES:

<small>[Introduction to Computer Vision - Udacity](https://classroom.udacity.com/courses/ud810){:target="_blank"}</small>

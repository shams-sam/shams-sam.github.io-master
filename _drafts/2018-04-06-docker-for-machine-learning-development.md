---
layout: post
title: "Docker for Machine Learning Developement"
categories: []
tags: [machine-learning, library, setup]
description: Docker is a computer program that performs operating-system-level virtualization also known as containerization. It is developed by Docker, Inc.
cover: "/assets/images/docker.jpg"
cover_source: "https://news-cdn.softpedia.com/images/news2/docker-1-13-1-implements-support-for-global-scoped-network-plugins-in-swarm-mode-512751-2.jpg"
comments: true
---

### Introduction

Containerization, also known as operating-system-level virtualization is a feature of the operating system in which the kernel allows the existence of multiple isolated user-space instances. Such instances may be called containers, partitions, virtualization engines (VEs), or jails. These look like real computers from the point of view of a program running in them.

### General Commands

```sh
# docker help 
docker
docker --help
docker <command-name> --help

# check docker version
docker --version

# check additional docker information
docker version
docker info

# check docker installation
docker run hello-world
```

### Images

* Docker **image** is an executable package that includes everything needed to run an application - code, libraries, environment variables, and configuration files.

```sh
# list images
docker image ls
docker images

# list all images on machine
docker image ls -a

# remove image from machine
docker image rm <image id>

# remove all images from this machine
docker image rm $(docker image ls -a -q)
```

### Container 

* Docker **container** is launched by running an image. Container is to image, what object is to class, i.e. container is a runtime instance of an image, or what the image becomes in memory when executed.

```sh
# list all running containers
docker container ls
docker ps

# list all containers
docker container ls --all
docker container ls -a

# list all containers in quite mode
docker container ls -aq

# docker container stop using container id
docker container stop <container-id>

# docker container kill using container id
docker container kill <container-id>

# docker container remove using container id
docker container rm <container-id>

# remove all containers
docker container rm $(docker container ls -a -q)
docker container ls -a | awk '{print "docker container rm ",$1}' | bash
```

### Container vs Virtual Machines

A container runs natively on Linux, sharing the kernel of the host machines with other containers. It only runs processes and occupies memory required by that executable, hence making it lightweight. 

On the contrary, a virtual machine runs a full guest operating system with virtual access to the host resources through a hypervisor. As a result, VMs provide an entire guest operating system and hence a lot more resources than what the application would actually need.

### Dockerfile

* **Dockerfile** defines the envrionment inside the container. Options like access to resources, disk drives etc. can be configured in the file. Post this, it can expected that the builds from this Dockerfile would behave the same wherever it runs.

```
# tensorflow nightly build base image with python 3
FROM tensorflow/tensorflow:nightly-py3

# set working directory
WORKDIR /development

# Copy the current directory contents into the destination directory
ADD . /development

# setup command line tools
RUN apt-get update && apt-get install -y \
    emacs \
    git \
    tmux \
    && \
apt-get clean && \
apt-get autoremove && \
rm -rf /var/lib/apt/list/*

# setup non-pip libraries
RUN ["apt-get", "install", "-y", "python-opencv"]

# install requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# expose ports for connection
# 6006 for tensorboard
# 8888 for jupyter notebook
EXPOSE 6006 8888

CMD ["/bin/bash"]
```

### Build from Dockerfile

* Once the Dockerfile is built and ready, it can be used to build and run the image by using the following commands,

```sh
# build image from Dockerfile
docker build -t <docker-image-name> .

# run container in interactive mode
docker run -it <image-name> <optional-command-to-overwrite-base-command>

# run container in detached mode
docker run -d <image-name>

# run container with port forwarding
docker run -p <host-port>:<container-port-exposed> <image-name>
```

### Docker Hub

* **Dockerhub** is the public repository of docker images where anyone can publish their images or pull images from. Image names on dockerhub have a name based on username of uploader, project name and tags based on properties of docker image - latest, nightly etc. A template would look like: **username/repository:tag**
* Once uploaded, all the images are publically available and can be availed on any system with connectivity, because if an image in not available locally, docker tries to pull it from the repository.

```sh
# login to dockerhub
docker login

# tag <image> for upload to registry
docker tag <image> username/repository:tag

# upload tagged image to registry
docker push username/repository:tag

# run image from a registry
docker run username/repository:tag

# pull tagged image from registry
docker pull username/repository:tag
```

## REFERENCES:

<small>[Docker - Get Started](https://docs.docker.com/get-started/){:target="_blank"}</small><br>

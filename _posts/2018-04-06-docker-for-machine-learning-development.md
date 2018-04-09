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
* An image is a read-only template with instructions or creating a docker container. It is possible for an image to be based on another base image, with additional customizations.
* Images are created using `Dockerfile` with a simple syntax the defines the steps involved in creating the image and run it. Each instruction in the `Dockerfile` creates a layer in the image and cached for the next build to speed up the process until the `Dockerfile` is changed itself.

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
* A container is defined by the image used to spawn it and the configuration options provided while creating or running it.
* When a container is removed, any changes in the state not stored in the persistent storage disappear. 

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

# attach to a running container
docker attach <container-id>
```

* `Ctrl-c` to kill the container.
* `Ctrl-p Ctrl-q` sequence to detach from an running container without killing it.

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

* Once the Dockerfile is ready, it can be used to build and run the image by using the following commands,

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

### Services

* Services are just **containers in production**. A service runs only one image, but along with it other information is also maintained - ports it should run on, number of replicas of the container to be maintained etc.
* Scaling a service basically means to change the number of containers running the particular piece of the software, assigning more compute resources to service in the process.
* Services allow the containers to be scaled across multiple docker daemons, which all work together in a swarm using multiple managers and workers. Each member of the swarm in a docker daemon communicating with other daemons using Docker APIs.

### Swarms

* A swarm is a group of machines that are running docker and joined into a cluster. Post definition of the cluster, all the commands are run on the entire cluster by a **swarm manager**. The machines in a cluster can be **virtual** or **physical**, and are termed as **nodes**.

**Swarm managers** can be configured in the compose file to use several different strategies to deploy new containers on the cluster,

* **Emptiest Node:** runs the container on the least utilized machine.
* **Global:** each machine gets exactly one instance of the specified node.

> Swarm managers are the only machines that can run commands on a swarm or allow new nodes to join as a worker in the cluster. Workers are only capable of donating the compute power but do not have the authority to rule any other machine using commands.

The moment swarm mode is switched on, the current machine becomes the swarm mananger and any commands run henceforth are run on the cluster by the swarm manager.

Setting up swarm,

* run `docker swarm init` to enable the swarm mode and make the current machine the swarm manager
* run `docker swarm join` on other machines to make them join the cluster
* run `docker swarm leave` on worker to leave the swarm

> Always run `docker swarm init` and `docker swarm join` with port 2377 (the swarm management port), or let it take the default port by leaving it empty.

> The machine IP addresses returned by `docker-machine ls` include port 2376, which is the Docker daemon port. Do not use this port or you may experience errors.

### Docker Machine

* Docker machine helps to start and manage the virtual machines in a cluster with ease.

```shell
# create a virtual machine
docker-machine create --driver virtualbox <vm-name>

# list as docker machines
docker-machine ls

# send command to docker machines
docker-machine ssh <vm-name> <command>

# to use the native ssh of docker
docker-machine --native-ssh ssh <vm-name> <command>

# configure shell to interact with docker machine
eval $(docker-machine env <vm-name>)
```

### Development Notes

There are various advantages of using docker for machine-learning development, 

* Library dependencies for complex machine learning libraries like tensorflow can be cumbersome to build sometimes. Using base images it is easier to load them. 
* Open-source docker images like [this](https://github.com/floydhub/dl-docker){:target="\_blank"} that give most of the libaries one may use, out of the box.
* It can help run mutliple CUDA/cuDNN versions on the same machines based on project requirements.
* Services and Swarms can help run experiments on bigger swarms using more compute power from mutliple workers that might not be easily available on a single machine.
* Once a system with basic dependencies is ready there would not be any overhead of reiterating the setup process on any other machine thanks to the excellent portability that docker offers, regardless of the host operating systems.

**Note: Updated copy of [Docker Setup Files](https://github.com/shams-sam/setups/tree/master/docker-setup){:target="\_blank"}**

### Stack

* A stack is the top-most hierarchy of the docker containerization. It is the group of related services that share dependencies, and can be orchestrated and scaled together. A single stack is capable of defining and coordinating the functionality of the entire application.

### Docker Architecture

* Docker uses a **client-server architecture**.
* The docker client speaks to docker daemon, which does the building, running and managing of the docker containers.
* The client and daemon communicate through REST APIs.

* **Docker Daemon:** listens to the APIs and manages the images, containers, networks and volumes. A daemon can communicate with other daemons and clients to manager the docker system.
* **Docker Client:** the primary way of communicating with the docker daemons. All the commands executed at the client are internally routed to `dockerd` which then executes them. A single docker client can communicate with many different docker daemons.
* **Docker Registries:** stores the docker images. For example, docker hub and docker cloud are the public repositories that anyone can use, and docker is configured by default to look for a image in docker hub in case it does not find local copy. It is also possible to run private registries. Commands like `docker pull` and `docker run` fetch the required images from the configured repositories.

### Miscellaneous

* Docker is written in **go language**.
* It is built to utilize the features of linux kernels in its full advantage.
* Docker provides container isolations using **namespaces**. On executing `docker run`, docker creates a set of namespaces for the container. Each aspect of the container is run on a different namespace and its access is limited to that namespace.

### Namespaces

Docker engine uses the following namespaces on linux,

* `pid` namespace: process isolation
* `net` namespace: managing network interfaces
* `ipc` namespace: managing access to IPC (InterProcess Communication) resources
* `mnt` namespace: managing filesystem mount points.
* `uts` namespace: isolating kernel and version identifiers.

### Contol Groups

* Control groups, `cgroups`, limits application to a specific set of resources.
* It helps docker engine to share available hardware resources among the running containers, and optionally enforce limits and constraints.

## REFERENCES:

<small>[Docker - Get Started](https://docs.docker.com/get-started/){:target="\_blank"}</small><br>

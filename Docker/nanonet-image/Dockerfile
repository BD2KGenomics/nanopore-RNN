# try to create docker image for nanonet


FROM ubuntu:14.04

MAINTAINER Andrew Bailey andbaile@ucsc.edu

RUN apt-get update  && apt-get install -y \
    build-essential \
    cmake \
    python \
    python-pip \
    python-dev \
    hdf5-tools \
    libhdf5-dev \
    hdf5-helpers \
    libhdf5-serial-dev \
    git \
    apt-utils\
    python-numpy


# RUN apt-get install python-pyopencl


# install nanonet from source
WORKDIR /docker
RUN git clone https://github.com/nanoporetech/nanonet
WORKDIR /docker/nanonet
RUN python setup.py install with2d


RUN mkdir /opt/nanonet
COPY wrapper.sh /opt/nanonet/

RUN mkdir /data
WORKDIR /data

ENTRYPOINT ["sh", "/opt/nanonet/wrapper.sh"]

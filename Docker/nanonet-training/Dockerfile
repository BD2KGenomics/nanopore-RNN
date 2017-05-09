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
    apt-utils \
    python-numpy \
    libboost-python1.54-dev \
    libboost-python1.54.0 \
    wget



RUN wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
RUN sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
RUN sudo apt-get update
RUN sudo apt-get -y install cuda-cudart-7-5 cuda-cublas-7-5 netcdf-bin libboost-all-dev

RUN wget https://github.com/nanoporetech/currennt/releases/download/v0.2-rc1-2/python-netcdf4_1.2.3-1_amd64.deb \
     https://github.com/nanoporetech/currennt/releases/download/v0.2-rc1-2/ont-currennt-0.2.1-3-trusty.deb

RUN dpkg -i python-netcdf4_1.2.3-1_amd64.deb; exit 0
RUN dpkg -i ont-currennt_0.2.1-2-trusty_amd64.deb; exit 0

RUN apt-get -fy install

RUN apt-get install -y netcdf-bin python-h5py cython

RUN mkdir /src
RUN mkdir /src/currennt

# COPY currennt-0.2-rc1.zip /src/currennt
# WORKDIR /src/current

# RUN unzip currennt-0.2-rc1.zip

# git clone https://github.com/nanoporetech/currennt.git

# install nanonet from source
WORKDIR /docker
RUN git clone https://github.com/nanoporetech/nanonet
WORKDIR /docker/nanonet

RUN pip install -e .[currennt]


RUN mkdir /opt/nanonet
COPY wrapper.sh /opt/nanonet/

RUN mkdir /data
WORKDIR /data

ENTRYPOINT ["sh", "/opt/nanonet/wrapper.sh"]

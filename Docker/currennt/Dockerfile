# try to create docker image for nanonettrain


FROM nvidia/cuda:7.5-devel-ubuntu14.04

MAINTAINER Andrew Bailey andbaile@ucsc.edu

RUN sudo apt-get update
RUN sudo apt-get install -y build-essential
RUN sudo apt-get install -y netcdf-bin
RUN sudo apt-get install -y libboost-all-dev
RUN sudo apt-get install -y python-pip
RUN sudo apt-get install -y python-h5py
RUN sudo apt-get install -y cython
RUN sudo apt-get install -y git
RUN sudo apt-get install -y cmake
# RUN apt-get install -y cuda; exit 0
RUN sudo apt-get install -y wget


ENV CUDA_BIN_PATH /usr/local/cuda/
ENV BASHRC "~/.bashrc"

# Install zlib
WORKDIR /docker
RUN wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4/zlib-1.2.8.tar.gz
RUN tar -xf zlib-1.2.8.tar.gz
WORKDIR /docker/zlib-1.2.8
RUN ./configure --prefix=/usr/local
RUN sudo make install

# Install HDF5
WORKDIR /docker
RUN wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4/hdf5-1.8.13.tar.gz
RUN tar -xf hdf5-1.8.13.tar.gz
WORKDIR /docker/hdf5-1.8.13
#
# prefix="/usr/local/hdf5-1.8.13"
# if [ $HDF5_DIR != $prefix ]; then
#     echo "Add HDF5_DIR=$prefix to .bashrc"
#     echo "" >> $BASHRC
#     echo "# HDF5 libraries for python" >> $BASHRC
#     echo export HDF5_DIR=$prefix  >> $BASHRC
# fi

RUN ./configure --enable-shared --enable-hl --prefix=$HDF5_DIR
RUN make -j 2
RUN sudo make install

# Install Netcdf
WORKDIR /docker
RUN wget http://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-4.1.3.tar.gz
RUN tar -xf netcdf-4.1.3.tar.gz
WORKDIR /docker/netcdf-4.1.3
# prefix="/usr/local/"
# if [ $NETCDF4_DIR != $prefix ]; then
#     echo "Add NETCDF4_DIR=$prefix to .bashrc"
#     echo "" >> $BASHRC
#     echo "# NETCDF4 libraries for python" >> $BASHRC
#     echo export NETCDF4_DIR=$prefix  >> $BASHRC
# fi
RUN CPPFLAGS=-I$HDF5_DIR/include LDFLAGS=-L$HDF5_DIR/lib ./configure --enable-netcdf-4 --enable-shared --enable-dap --prefix=$NETCDF4_DIR
# make check
RUN make
RUN sudo make install

# Install Currennt
WORKDIR /docker
RUN git clone https://github.com/nanoporetech/currennt.git
WORKDIR /docker/currennt
RUN mkdir build
WORKDIR /docker/currennt/build
RUN cmake ..
RUN make

WORKDIR /docker
RUN git clone https://github.com/nanoporetech/nanonet
WORKDIR /docker/nanonet

RUN sudo apt-get install -y libhdf5-dev python-numpy

# RUN pip install -e .[currennt]
#
#
# RUN mkdir /opt/nanonet
# COPY wrapper.sh /opt/nanonet/

RUN mkdir /data
WORKDIR /data

ENTRYPOINT ["sh", "/opt/nanonet/wrapper.sh"]
